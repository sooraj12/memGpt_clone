import { useCallback, useState } from "react";
import { useGlobalStore } from "../store/useGlobalStore";
import { useShallow } from "zustand/react/shallow";

export function useOption(groupID, optionID, chatID) {
  const { chatManager } = useGlobalStore(
    useShallow((state) => ({
      chatManager: state.chats.chatManager,
    }))
  );
  const [value, setValue] = useState(
    chatManager.options.getValidatedOption(groupID, optionID, chatID)
  );

  const setOptionValue = useCallback(
    (value) => {
      chatManager.options.setOption(groupID, optionID, value, chatID);
    },
    [groupID, optionID, chatID]
  );

  const option = chatManager.options.findOption(groupID, optionID);

  return [
    value,
    setOptionValue,
    typeof option.renderProps === "function"
      ? option.renderProps(value, chatManager.options, chatManager)
      : option.renderProps,
  ];
}
