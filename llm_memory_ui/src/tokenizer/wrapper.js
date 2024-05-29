// import tokenizer from "./worker";

// const worker = new Worker(new URL(tokenizer, import.meta.url), {
//   type: "module",
// });

const worker = {};

export async function runChatTrimmer(messages, options) {
  return worker.runChatTrimmer(messages, options);
}

export async function countTokens(messages) {
  return await worker.countTokensForMessages(messages);
}
