import styled from "@emotion/styled";
import { Button, ActionIcon, Textarea, Loader } from "@mantine/core";
import { getHotkeyHandler, useMediaQuery } from "@mantine/hooks";
import { useCallback, useMemo } from "react";
import { useLocation } from "react-router-dom";
import { useGlobalStore } from "../store/useGlobalStore";
import { useShallow } from "zustand/react/shallow";

const Container = styled.div`
  background: #292933;
  border-top: thin solid #393933;
  padding: 1rem 1rem 0.5rem 1rem;

  .inner {
    max-width: 50rem;
    margin: auto;
    text-align: right;
  }

  .settings-button {
    margin: 0.5rem -0.4rem 0.5rem 1rem;
    font-size: 0.7rem;
    color: #999;
  }
`;

function MessageInput(props) {
  const { setMessage, message, sendMessage, generating } = useGlobalStore(
    useShallow((state) => ({
      setMessage: state.chats.setMessage,
      message: state.chats.msg,
      sendMessage: state.chats.sendMessage,
      generating: state.chats.generating,
    }))
  );
  const location = useLocation();
  const isHome = location.pathname === "/";

  const hasVerticalSpace = useMediaQuery("(min-height: 1000px)");

  const submitOnEnter = true;

  const onChange = useCallback(
    (e) => {
      setMessage(e.target.value);
    },
    [setMessage]
  );

  const onSubmit = useCallback(async () => {
    setMessage("");

    await sendMessage(message?.trim());
  }, [message, setMessage, sendMessage]);

  const blur = useCallback(() => {
    document.querySelector("#message-input")?.blur();
  }, []);

  const rightSection = useMemo(() => {
    return (
      <div
        style={{
          opacity: "0.8",
          paddingRight: "0.5rem",
          display: "flex",
          justifyContent: "flex-end",
          alignItems: "center",
          width: "100%",
        }}
      >
        {generating && (
          <>
            <Button variant="subtle" size="xs" compact onClick={() => {}}>
              Cancel
            </Button>
            <Loader size="xs" style={{ padding: "0 0.8rem 0 0.5rem" }} />
          </>
        )}
        {!generating && (
          <>
            <ActionIcon size="xl" onClick={onSubmit}>
              <i className="fa fa-paper-plane" style={{ fontSize: "90%" }} />
            </ActionIcon>
          </>
        )}
      </div>
    );
  }, [onSubmit, generating]);

  const hotkeyHandler = useMemo(() => {
    const keys = [
      ["Escape", blur, { preventDefault: true }],
      ["ctrl+Enter", onSubmit, { preventDefault: true }],
    ];
    if (submitOnEnter) {
      keys.unshift(["Enter", onSubmit, { preventDefault: true }]);
    }
    const handler = getHotkeyHandler(keys);
    return handler;
  }, [onSubmit, blur, submitOnEnter]);

  return (
    <Container>
      <div className="inner">
        <Textarea
          disabled={props.disabled || generating}
          id="message-input"
          autosize
          minRows={hasVerticalSpace || isHome ? 3 : 2}
          maxRows={12}
          placeholder={"Ask a question..."}
          value={message}
          onChange={onChange}
          rightSection={rightSection}
          rightSectionWidth={generating ? 100 : 55}
          onKeyDown={hotkeyHandler}
        />
      </div>
    </Container>
  );
}

export { MessageInput };
