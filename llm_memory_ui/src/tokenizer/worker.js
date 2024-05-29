import { expose } from "comlink";
// import * as methods from ".";
// import { ChatHistoryTrimmer } from "./chat-history-trimmer";

export function runChatTrimmer(messages, options) {
  //   const trimmer = new ChatHistoryTrimmer(messages, options);
  //   return trimmer.process();
}

export function countTokensForText(text) {
  // return methods.countTokensForText(text);
  return 1000;
}

export function countTokensForMessages(messages) {
  // return methods.countTokensForMessages(messages);
  return 1000;
}

export default function tokenizer(txt) {
  return txt;
}

expose({ runChatTrimmer, countTokensForText, countTokensForMessages });
