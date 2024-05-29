import { globalOptions } from "../config/globalOptions";

class OptionsManager {
  constructor(yDoc, pluginMetadata) {
    this.optionGroups = [...globalOptions, ...pluginMetadata];
  }

  getValidatedOption(groupID, optionID, chatID) {
    return this.getOption(groupID, optionID, chatID, true);
  }

  getOption(groupID, optionID, chatID, validate) {
    const option = this.findOption(groupID, optionID);

    return option.defaultValue;
  }

  findOption(groupID, optionID) {
    const group = this.optionGroups.find((group) => group.id === groupID);
    const option = group?.options.find((option) => option.id === optionID);

    return option;
  }

  getAllOptions(groupID, chatID) {
    let options = [];
    const group = this.optionGroups.find((group) => group.id === groupID);
    group?.options.forEach((option) => {
      options[option.id] = this.getOption(groupID, option.id, chatID);
    });

    return options;
  }
}

export { OptionsManager };
