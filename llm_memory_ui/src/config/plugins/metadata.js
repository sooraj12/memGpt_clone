import { registeredPlugins } from "./registeredPlugins";

export const pluginMetadata = registeredPlugins.map((p) => new p().describe());

export const pluginIDs = pluginMetadata.map((d) => d.id);

export function getPluginByName(name) {
  return registeredPlugins.find((p) => new p().describe().name === name);
}
