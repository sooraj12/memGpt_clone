import { OptionsManager } from "./optionsManager";
import { pluginMetadata } from "../config/plugins/metadata";

class ChatManager {
  constructor() {
    console.log("initializing chat manager");

    this.doc = this.attachYDoc("anonymous");
  }


  attachYDoc() {
    this.doc = "";

    this.options = new OptionsManager(this.doc, pluginMetadata);

    return this.doc;
  }

  getQuickSettings() {
    const options = this.options.getAllOptions("quick-settings");
    return Object.keys(options)
      .filter((key) => options[key])
      .map((key) => {
        const groupID = key.split("--")[0];
        const optionID = key.split("--")[1];
        return {
          groupID,
          option: this.options.findOption(groupID, optionID),
        };
      })
      .filter((o) => !!o.option);
  }
}

export { ChatManager };
