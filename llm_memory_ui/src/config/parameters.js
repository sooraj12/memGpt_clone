export const parameterOptions = {
  id: "parameters",
  options: [
    {
      id: "model",
      defaultValue: "llama3",
      resettable: false,
      scope: "user",
      displayOnSettingsScreen: "chat",
      displayAsSeparateSection: true,
      displayInQuickSettings: {
        name: "Model",
        displayByDefault: true,
        label: (value) => value,
      },
      renderProps: (value, options, context) => ({
        type: "select",
        label: "Model",
        description:
          value?.includes("32") &&
          "Note: This model will only work if your OpenAI account has been granted you have been given access to it. <a>Request access here.</a>",
        options: [
          {
            label: "LLAMA3 70B instruct",
            value: "llama3:70b",
          },
        ],
      }),
    },
    {
      id: "temperature",
      defaultValue: 0.5,
      resettable: true,
      scope: "chat",
      displayOnSettingsScreen: "chat",
      displayAsSeparateSection: true,
      displayInQuickSettings: {
        name: "Temperature",
        displayByDefault: false,
        label: (value) => "Temperature: " + value.toFixed(1),
      },
      renderProps: (value, options, context) => ({
        type: "slider",
        label: "Temperature: " + value.toFixed(1),
        min: 0,
        max: 1,
        step: 0.1,
        description:
          "The temperature parameter controls the randomness of the AI's responses. Lower values will make the AI more predictable, while higher values will make it more creative.",
      }),
    },
  ],
};
