import { countTokens, runChatTrimmer } from "../../tokenizer/wrapper";

export const systemPrompt = `
Please read the following exchange and write a short, concise title describing the topic (in the user's language).
If there is no clear topic for the exchange, respond with: N/A
`.trim();

export const systemPromptForLongExchanges = `
Please read the following exchange and write a short, concise title describing the topic (in the user's language).
`.trim();

const userPrompt = (messages) => {
  return (
    messages
      .map((m) => `${m.role.toLocaleUpperCase()}:\n${m.content}`)
      .join("\n===\n") + "\n===\nTitle:"
  );
};

export class TitlePlugin {
  describe() {
    return {
      id: "titles",
      name: "Title Generator",
      options: [],
    };
  }

  async postprocessModelOutput(message, contextMessages, parameters, done) {
    if (done && !this.context?.getCurrentChat().title) {
      (async () => {
        let messages = [
          ...contextMessages.filter(
            (m) => m.role === "user" || m.role === "assistant"
          ),
          message,
        ];

        const tokens = await countTokens(messages);

        messages = await runChatTrimmer(messages, {
          maxTokens: 1024,
          preserveFirstUserMessage: true,
          preserveSystemPrompt: false,
        });

        messages = [
          {
            role: "system",
            content:
              tokens.length > 512 ? systemPromptForLongExchanges : systemPrompt,
          },
          {
            role: "user",
            content: userPrompt(messages),
          },
        ];

        const output = await this.context?.createChatCompletion(messages, {
          model: "LLAMA3",
          temperature: 0,
        });

        if (!output || output === "N/A") {
          return;
        }

        this.context?.setChatTitle(output);
      })();
    }
    return message;
  }
}
