import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { settingsStore } from "./settings";
import { chatsStore } from "./chats";

const useGlobalStore = create(
  devtools(
    immer((...utils) => ({
      ...settingsStore(...utils),
      ...chatsStore(...utils),
    }))
  )
);

export { useGlobalStore };
