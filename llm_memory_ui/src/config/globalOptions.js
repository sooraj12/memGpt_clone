import { parameterOptions } from "./parameters";
import { pluginMetadata } from "./plugins/metadata";
import { autoScrollOptions, inputOptions, markdownOptions } from "./ui";

export const globalOptions = [
  autoScrollOptions,
  parameterOptions,
  inputOptions,
  markdownOptions,
];

const optionsForQuickSettings = [];
[...globalOptions, ...pluginMetadata].forEach((plugin) => {
  plugin.options.forEach((option) => {
    if (option.displayInQuickSettings) {
      optionsForQuickSettings.push({
        id: plugin.id + "--" + option.id,
        defaultValue: !!option.displayInQuickSettings?.displayByDefault,
        displayOnSettingsScreen: "ui",
        displayAsSeparateSection: false,
        renderProps: {
          type: "checkbox",
          label: option.displayInQuickSettings?.name || option.id,
        },
      });
    }
  });
});

const quickSettings = {
  id: "quick-settings",
  name: "Quick Settings",
  options: optionsForQuickSettings,
};

globalOptions.push(quickSettings);
