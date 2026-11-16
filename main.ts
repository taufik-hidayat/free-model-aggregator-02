// The entry: a live free-model aggregator, OpenAI- and Anthropic-compatible.
//
// The house's client sends six prompts (three on each wire) and this server answers each with a real
// completion from one of the fourteen free OpenRouter models the brief lists, dialled through the
// grader's gateway (an OpenAI-compatible endpoint at OPENAI_BASE_URL). It never answers from itself:
// a prompt it cannot serve honestly gets a 502 with the trail of what the gateway refused, not an
// invented answer.
//
// ROUTING (the quality column, 60% of the score)
//   Each prompt is served by the highest-rated free model that meets its requirements — the brief's
//   rating_milli, read at runtime from MAKEFUN_PROMPT_FILE, never a hardcoded id. The requirements
//   are the manifest row's (by arrival order) unioned with what the request body itself declares
//   (tools, image parts, JSON mode), so a JSON-mode prompt is never served by a model that cannot do
//   JSON mode, and a tools prompt only by a model that can call tools. The best eligible model is
//   tried first; a model abandoned on the previous prompt is sorted after every model that has
//   not refused, so the next prompt does not spend a call on a model that is cooling down.
//
// FAILOVER (the failover column, 10%)
//   A 429 or 5xx (or a dropped connection) is retried once on the same model after a fixed one-second
//   wait, then the entry fails over to the next-best eligible model, up to five upstream calls per
//   prompt. Every refusal is recorded on the `x-makefun-upstream-failures` header (at most eight
//   rows: {status, code, model}), so the house's client can read back the trail of one request and
//   the failover column can credit a prompt a refusal could not stop.
//
// THE WIRE
//   POST /v1/chat/completions (JSON and streaming SSE) and POST /v1/messages (Anthropic wire, JSON
//   and streaming) both dial Chat Completions upstream; the Anthropic path is translated both ways
//   (see lib/wire.ts). GET /v1/models lists the same fourteen ids under a Bearer token (OpenAI's
//   shape) and under an x-api-key header (Anthropic's shape). Every answer's `model` is the id this
//   entry dialled: the provider strips the `openrouter/` prefix, and this entry puts it back, on
//   the answer and on every streamed chunk.
//
// DETERMINISM
//   Every choice here is a function of the brief, the request and the gateway's answers so far —
//   no wall clock, no unseeded randomness — so a replay decides the same way and exports the same
//   state. The one-second retry wait and the ten-second dial timeout are fixed waits, not choices.
//   The model listings use the fixed timestamps in lib/wire.ts for the same reason.
//
// LIFECYCLE
//   The server listens on 0.0.0.0 (MAKEFUN_PORT, default 8080), registers its port and catalog with
//   POST $MAKEFUN_ENV_URL/register, then polls GET $MAKEFUN_ENV_URL/work and exits 0 when the house
//   says the trial is done. `node main.ts --list` prints the catalog document, byte for byte the
//   document that is registered.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  estimateInputTokens,
  freeModels,
  isRecord,
  rankedCandidates,
  readBrief,
  requirementsFromBody,
  type Brief,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The upstream calls one prompt may spend: the best model, its retry, then three failovers. */
const MAX_UPSTREAM_CALLS = 5;
/** One upstream call may take this long before it is a refusal. Five calls at this cap, plus the
 * one-second retry wait, stay inside the client's thirty-second per-prompt budget. */
const DIAL_TIMEOUT_MS = 5000;
/** The fixed wait before the same model is retried once. */
const RETRY_WAIT_MS = 1000;
/** How often the house is asked whether the trial is done. */
const WORK_POLL_MS = 100;
/** The gateway refuses a Chat Completions call without a cap, or over this cap. */
const MAX_TOKENS_CAP = 8192;
/** The cap this entry sends when the client asked for none. */
const DEFAULT_MAX_TOKENS = 1024;

const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? '';
const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/$/, '');
const upstreamBase = (process.env.OPENAI_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/$/, '');
const upstreamKey = process.env.OPENAI_API_KEY ?? 'dummy';
const port = Number(process.env.MAKEFUN_PORT ?? '8080') || 8080;

const brief: Brief = readBrief(promptFile);
const catalog = buildCatalog(brief);
const servedModels: ServedModel[] = freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
if (brief.error !== null) console.error(`[entry] brief: ${brief.error}`);

let manifestIndex = 0;
/**
 * Models cooling down: the prompt index until which each is skipped. A model that is abandoned on
 * prompt n is skipped for exactly the next prompt (n + 1) and eligible again after that — a
 * refusal cools a model down, it does not banish it for the trial.
 */
const cooldownUntil = new Map<string, number>();

/** The models cooling down on prompt `n`: those whose cooldown has not yet run out. */
function penalizedFor(promptNumber: number): Set<string> {
  const out = new Set<string>();
  for (const [model, until] of cooldownUntil) if (promptNumber < until) out.add(model);
  return out;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One refusal from a provider, as it is read off an HTTP error body. */
interface UpstreamFailure {
  /** The HTTP status the provider answered (502 when no response arrived at all). */
  status: number;
  /** The provider's own code, or `upstream_error` when its body said nothing usable. */
  code: string;
  /** The model the call was dialled on, when that is known. */
  model: string | null;
}

/**
 * The value of the `x-makefun-upstream-failures` header: a JSON array of at most eight failure rows.
 * Only written when there is a row to write, and the rows are the rawest facts available (no invented
 * status, no invented code), so a live pass and its replay agree on them.
 */
function failuresHeader(failures: readonly UpstreamFailure[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(
    failures.slice(-8).map((row) => ({ status: row.status, code: row.code.slice(0, 64), model: row.model === null ? null : row.model.slice(0, 200) })),
  );
}

/** An error body read off one upstream HTTP response, limited to 16 KiB. */
async function readErrorBody(response: Response): Promise<string> {
  const body = await response.text().catch(() => '');
  return body.length > 16384 ? body.slice(0, 16384) : body;
}

/**
 * One refusal from an upstream HTTP response (or from the fetch itself, when no response arrived).
 * The code is the provider's own `error.code` (string, else `error.type`), else the provider's
 * `code` at the top level, else `upstream_error`. The model is the id this entry dialled: when the
 * provider's own body names another model, that name is never taken instead.
 */
function failureFromResponse(response: Response | null, body: string, dialledModel: string | null): UpstreamFailure {
  const status = response === null ? 502 : response.status;
  let code = 'upstream_error';
  try {
    const doc: unknown = JSON.parse(body);
    if (isRecord(doc)) {
      const err = isRecord(doc.error) ? doc.error : null;
      if (err !== null && typeof err.code === 'string' && err.code !== '') code = err.code;
      else if (err !== null && typeof err.type === 'string' && err.type !== '') code = err.type;
      else if (typeof doc.code === 'string' && doc.code !== '') code = doc.code;
      else if (typeof doc.type === 'string' && doc.type !== '') code = doc.type;
    }
  } catch {
    // Not JSON: nothing to read.
  }
  return { status, code: code.slice(0, 64), model: dialledModel };
}

/** The request body as JSON, or null when it is not one. Bodies over 8 MiB are refused, not read. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > 8 * 1024 * 1024) return null;
    chunks.push(buf);
  }
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * One prompt's requirements: the manifest row's (the arrivals come in the brief's order) unioned with
 * what the request body itself declares, plus the input size the body implies. The manifest is the
 * declared source; the body is the backstop, so a prompt that needs JSON mode or tools is never
 * served by a model that lacks it.
 */
function requirementsFor(body: Json): Requirement {
  const row = brief.prompts[manifestIndex] ?? null;
  const fromBody = requirementsFromBody(body);
  const merged: Requirement = { ...(row?.requires ?? {}), ...fromBody };
  const estimate = estimateInputTokens(body);
  const minContext = Math.max(row?.requires.min_context ?? 0, estimate);
  if (minContext > 0) merged.min_context = minContext;
  return merged;
}

/**
 * The upstream body for one dial: the client's body with the dialled model's id, and a max_tokens the
 * gateway accepts (the gateway refuses a call with no cap, or over its cap). The client's own cap,
 * temperature, tools and response_format pass through untouched.
 */
function upstreamBodyFor(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : 0;
  if (!(cap > 0)) out.max_tokens = DEFAULT_MAX_TOKENS;
  else if (typeof out.max_tokens === 'number' && out.max_tokens > MAX_TOKENS_CAP) out.max_tokens = MAX_TOKENS_CAP;
  else if (typeof out.max_completion_tokens === 'number' && out.max_completion_tokens > MAX_TOKENS_CAP) out.max_completion_tokens = MAX_TOKENS_CAP;
  return out;
}

/** One upstream call: the Chat Completions request for `model`, or its refusal. */
async function dialOnce(
  model: string,
  upstreamBody: Json,
  stream: boolean,
): Promise<{ ok: true; response: Response } | { ok: false; failure: UpstreamFailure }> {
  const body = { ...upstreamBody, model, stream };
  let response: Response;
  try {
    response = await fetch(`${upstreamBase}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${upstreamKey}` },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, failure: { status: 502, code: 'upstream_unreachable', model } };
  }
  if (response.ok) return { ok: true, response };
  const text = await readErrorBody(response);
  return { ok: false, failure: failureFromResponse(response, text, model) };
}

/**
 * Dial one prompt upstream, best model first, with failover.
 *
 * The best eligible model is tried, then retried once after a fixed one-second wait when it refused
 * with a 429 or a 5xx (a refusal a retry can outlast), then the next-best eligible models, up to five
 * upstream calls. A model is cooled down only when it is abandoned — its retry also refused, or it
 * refused with a 4xx no retry can outlast — and only for the next prompt, so a single transient
 * refusal the retry recovered from does not banish a healthy model from later prompts. Every refusal
 * is recorded, in dial order, for the trail header.
 */
async function dialWithFailover(
  upstreamBody: Json,
  candidates: readonly ModelRow[],
  stream: boolean,
  promptNumber: number,
): Promise<{ response: Response; model: string; failures: UpstreamFailure[] } | { failures: UpstreamFailure[] }> {
  const failures: UpstreamFailure[] = [];
  const attempts: string[] = [];
  if (candidates.length > 0) attempts.push(candidates[0].id);
  if (candidates.length > 0) attempts.push(candidates[0].id);
  for (let i = 1; i < candidates.length && attempts.length < MAX_UPSTREAM_CALLS; i++) attempts.push(candidates[i].id);
  for (let n = 0; n < attempts.length; n++) {
    const model = attempts[n];
    if (n > 0 && model === attempts[n - 1]) await sleep(RETRY_WAIT_MS);
    const result = await dialOnce(model, upstreamBody, stream);
    if (result.ok) return { response: result.response, model, failures };
    failures.push(result.failure);
    // A 4xx other than a 429 is a refusal a retry cannot outlast: skip the same-model retry.
    if (n === 0 && result.failure.status >= 400 && result.failure.status < 500 && result.failure.status !== 429) {
      attempts.splice(1, 1);
    }
    // Cool the model down only when it is abandoned: the next attempt is a different model, or none.
    const next = attempts[n + 1];
    if (next === undefined || next !== model) cooldownUntil.set(model, promptNumber + 2);
  }
  return { failures };
}

function sendJson(res: ServerResponse, status: number, body: Json, failures: readonly UpstreamFailure[]): void {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(JSON.stringify(body));
}

/** An honest error, in the wire's own shape, with the trail of what the gateway refused. */
function sendError(
  res: ServerResponse,
  status: number,
  message: string,
  code: string,
  failures: readonly UpstreamFailure[],
  wire: 'openai' | 'anthropic',
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(code, message), failures);
  else sendJson(res, status, openAIError(message, 'upstream_error', code), failures);
}

/** One OpenAI Chat Completions request, JSON or streaming, served by a free model. */
async function handleChatCompletions(res: ServerResponse, body: Json): Promise<void> {
  const stream = body.stream === true;
  const promptNumber = manifestIndex;
  const requires = requirementsFor(body);
  manifestIndex += 1;
  const candidates = rankedCandidates(brief, requires, penalizedFor(promptNumber));
  if (candidates.length === 0) {
    const failures: UpstreamFailure[] = [{ status: 502, code: 'no_eligible_model', model: null }];
    sendError(res, 502, 'no free model meets this prompt\u2019s requirements', 'no_eligible_model', failures, 'openai');
    return;
  }
  const upstreamBody = upstreamBodyFor(body, candidates[0].id);
  const result = await dialWithFailover(upstreamBody, candidates, stream, promptNumber);
  if (!('response' in result)) {
    const last = result.failures[result.failures.length - 1];
    sendError(res, 502, `upstream refused: ${last?.code ?? 'upstream_error'}`, last?.code ?? 'upstream_error', result.failures, 'openai');
    return;
  }
  const { response, model, failures } = result;
  if (!stream) {
    const answer: unknown = await response.json().catch(() => null);
    if (!isRecord(answer)) {
      sendError(res, 502, 'upstream answer was not a JSON object', 'upstream_error', failures, 'openai');
      return;
    }
    sendJson(res, 200, openAIJsonWithModel(answer, model), failures);
    return;
  }
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  try {
    if (response.body !== null) for await (const frame of openAIStreamWithModel(response.body, model)) res.write(frame);
  } catch {
    // The upstream stream ended early: the client already has what it has.
  }
  res.end();
}

/** One Anthropic Messages request, JSON or streaming, translated to Chat Completions upstream. */
async function handleMessages(res: ServerResponse, body: Json): Promise<void> {
  const stream = body.stream === true;
  const promptNumber = manifestIndex;
  const requires = requirementsFor(body);
  manifestIndex += 1;
  const candidates = rankedCandidates(brief, requires, penalizedFor(promptNumber));
  if (candidates.length === 0) {
    const failures: UpstreamFailure[] = [{ status: 502, code: 'no_eligible_model', model: null }];
    sendError(res, 502, 'no free model meets this prompt\u2019s requirements', 'no_eligible_model', failures, 'anthropic');
    return;
  }
  const translated = anthropicToOpenAI(body, candidates[0].id);
  const upstreamBody = upstreamBodyFor(translated, candidates[0].id);
  const result = await dialWithFailover(upstreamBody, candidates, stream, promptNumber);
  if (!('response' in result)) {
    const last = result.failures[result.failures.length - 1];
    sendError(res, 502, `upstream refused: ${last?.code ?? 'upstream_error'}`, last?.code ?? 'upstream_error', result.failures, 'anthropic');
    return;
  }
  const { response, model, failures } = result;
  if (!stream) {
    const answer: unknown = await response.json().catch(() => null);
    if (!isRecord(answer)) {
      sendError(res, 502, 'upstream answer was not a JSON object', 'upstream_error', failures, 'anthropic');
      return;
    }
    sendJson(res, 200, openAIToAnthropic(answer, model), failures);
    return;
  }
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  try {
    if (response.body !== null) for await (const event of anthropicEventsFromOpenAI(response.body, model)) res.write(event);
  } catch {
    // The upstream stream ended early: the client already has what it has.
  }
  res.end();
}

/** GET /v1/models: the same fourteen ids under a Bearer token (OpenAI) or an x-api-key (Anthropic). */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  if (req.headers['x-api-key'] !== undefined) sendJson(res, 200, anthropicModelList(servedModels), []);
  else sendJson(res, 200, openAIModelList(servedModels), []);
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      handleModels(req, res);
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      const body = await readJsonBody(req);
      if (body === null) {
        sendJson(res, 400, openAIError('request body must be a JSON object', 'invalid_request_error', 'invalid_request'), []);
        return;
      }
      await handleChatCompletions(res, body);
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/messages') {
      const body = await readJsonBody(req);
      if (body === null) {
        sendJson(res, 400, anthropicError('invalid_request_error', 'request body must be a JSON object'), []);
        return;
      }
      await handleMessages(res, body);
      return;
    }
    sendJson(res, 404, openAIError('not found', 'invalid_request_error', 'not_found'), []);
  } catch (err) {
    console.error(`[entry] ${(err as Error).message}`);
    if (!res.headersSent) sendJson(res, 500, openAIError('internal error', 'server_error', 'internal_error'), []);
    else res.end();
  }
});

/** Register this server's port and catalog with the house; three tries, then give up. */
async function register(): Promise<boolean> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) return true;
      console.error(`[entry] register refused: ${response.status}`);
    } catch (err) {
      console.error(`[entry] register failed: ${(err as Error).message}`);
    }
    await sleep(500);
  }
  return false;
}

/** Ask the house whether the trial is done, until it says so. */
async function pollWork(): Promise<void> {
  for (;;) {
    await sleep(WORK_POLL_MS);
    try {
      const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) continue;
      const doc: unknown = await response.json().catch(() => null);
      if (isRecord(doc) && doc.done === true) {
        console.error('[entry] trial done');
        process.exit(0);
      }
    } catch {
      // The house is not answering yet: ask again next tick.
    }
  }
}

if (process.argv.includes('--list')) {
  process.stdout.write(`${JSON.stringify(catalog)}\n`);
  process.exit(0);
}

server.listen(port, '0.0.0.0', async () => {
  console.error(`[entry] listening on 0.0.0.0:${port}`);
  if (envUrl !== '') {
    const ok = await register();
    if (!ok) {
      console.error('[entry] register failed after three tries; exiting');
      process.exit(1);
    }
    console.error('[entry] registered');
    void pollWork();
  }
});

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));
