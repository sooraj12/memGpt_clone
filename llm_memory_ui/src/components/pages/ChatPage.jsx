import React, { Suspense, useCallback } from "react";
import styled from "@emotion/styled";
import { useEffect } from "react";
import { Loader } from "@mantine/core";
import { Page } from "../Page";
import { Message } from "../Message";
import { useGlobalStore } from "../../store/useGlobalStore";
import { useShallow } from "zustand/react/shallow";

const Messages = styled.div`
  @media (min-height: 30em) {
    max-height: 100%;
    flex-grow: 1;
    overflow-y: auto;
  }
  display: flex;
  flex-direction: column;
`;

const EmptyMessage = styled.div`
  flex-grow: 1;
  padding-bottom: 5vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  font-family: "Work Sans", sans-serif;
  line-height: 1.7;
  gap: 1rem;
  min-height: 10rem;
`;

function ChatPage() {
  const { generating, history } = useGlobalStore(
    useShallow((state) => ({
      generating: state.chats.generating,
      history: state.chats.history,
    }))
  );

  const autoScrollWhenOpeningChat = true;
  const autoScrollWhileGenerating = true;

  useEffect(() => {
    const shouldScroll = autoScrollWhenOpeningChat;

    if (!shouldScroll) {
      return;
    }

    const container = document.querySelector("#messages");
    const messages = document.querySelectorAll("#messages .message");

    if (messages.length) {
      const latest = messages[messages.length - 1];
      const offset = Math.max(0, latest.offsetTop - 100);
      setTimeout(() => {
        container?.scrollTo({ top: offset, behavior: "smooth" });
      }, 100);
    }
  }, [autoScrollWhenOpeningChat]);

  const autoScroll = useCallback(() => {
    if (generating && autoScrollWhileGenerating) {
      const container = document.querySelector("#messages");
      container?.scrollTo({ top: 999999, behavior: "smooth" });
      container?.parentElement?.scrollTo({ top: 999999, behavior: "smooth" });
    }
  }, [generating, autoScrollWhileGenerating]);

  useEffect(() => {
    const timer = setInterval(() => autoScroll(), 1000);
    return () => {
      clearInterval(timer);
    };
  }, [autoScroll]);

  return (
    <Page id={"landing"}>
      <Suspense
        fallback={
          <Messages id="messages">
            <EmptyMessage>
              <Loader variant="dots" />
            </EmptyMessage>
          </Messages>
        }
      >
        <Messages id="messages">
          {/* {!activeChat.isLoading && ( */}
          <div style={{ paddingBottom: "4.5rem" }}>
            {history.map((message, i) => (
              <Message
                key={message.id}
                message={message}
                last={i === history.length - 1}
              />
            ))}
          </div>
          {/* )} */}
          {/* {activeChat.isLoading && (
            <EmptyMessage>
              <Loader variant="dots" />
            </EmptyMessage>
          )} */}
        </Messages>
      </Suspense>
    </Page>
  );
}

export { ChatPage };
