import { countTokens, runChatTrimmer } from "../../tokenizer/wrapper";

export class ContextTrimmerPlugin {
  describe() {
    return {
      id: "context-trimmer",
      name: "Message Context",
      options: [
        {
          id: "maxTokens",
          displayOnSettingsScreen: "chat",
          defaultValue: 8000,
          scope: "chat",
          renderProps: (value, options) => ({
            label: `Include a maximum of ${value} tokens`,
            type: "slider",
            min: 512,
            max: 16000,
            step: 64,
          }),
          validate: (value, options) => {
            const max = 16000;
            return value <= max;
          },
          displayInQuickSettings: {
            name: "Max Tokens",
            displayByDefault: true,
            label: (value) => `Max tokens: ${value}`,
          },
        },
      ],
    };
  }

  async preprocessModelInput(messages, parameters) {
    const before = await countTokens(messages);

    const options = this.options;

    const trimmed = await runChatTrimmer(messages, {
      maxTokens: options?.maxTokens ?? 2048,
      nMostRecentMessages: options?.maxMessages ?? undefined,
      preserveFirstUserMessage: options?.preserveFirstUserMessage || true,
      preserveSystemPrompt: options?.preserveSystemPrompt || true,
    });

    const after = await countTokens(trimmed);

    const diff = after - before;
    console.log(`[context trimmer] trimmed ${diff} tokens from context`);

    return {
      messages: trimmed,
      parameters,
    };
  }
}
