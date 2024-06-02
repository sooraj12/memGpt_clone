import { useCallback } from "react";
import styled from "@emotion/styled";
import { Burger } from "@mantine/core";
import { useHotkeys } from "@mantine/hooks";

const HeaderContainer = styled.div`
  display: flex;
  flex-shrink: 0;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  min-height: 2.618rem;
  background: rgba(0, 0, 0, 0);
  font-family: "Work Sans", sans-serif;

  &.shaded {
    background: rgba(0, 0, 0, 0.2);
  }

  h1 {
    @media (max-width: 40em) {
      width: 100%;
      order: -1;
    }

    font-family: "Work Sans", sans-serif;
    font-size: 1rem;
    line-height: 1.3;

    animation: fadein 0.5s;
    animation-fill-mode: forwards;

    strong {
      font-weight: bold;
      white-space: nowrap;
    }

    span {
      display: block;
      font-size: 70%;
      white-space: nowrap;
    }

    @keyframes fadein {
      from {
        opacity: 0;
      }
      to {
        opacity: 1;
      }
    }
  }

  h2 {
    margin: 0 0.5rem;
    font-size: 1rem;
    white-space: nowrap;
  }

  .spacer {
    flex-grow: 1;
  }

  i {
    font-size: 90%;
  }

  i + span,
  .mantine-Button-root span.hide-on-mobile {
    @media (max-width: 40em) {
      position: absolute;
      left: -9999px;
      top: -9999px;
    }
  }

  .mantine-Button-root {
    @media (max-width: 40em) {
      padding: 0.5rem;
    }
  }
`;

// function HeaderButton({ icon, onClick, children }) {
//   return (
//     <Button size="xs" variant="subtle" onClick={onClick}>
//       {icon && <i className={"fa fa-" + icon} />}
//       {children && <span>{children}</span>}
//     </Button>
//   );
// }

function Header() {
  const onNewChat = useCallback(async () => {
    setTimeout(() => document.querySelector("#message-input")?.focus(), 100);
  }, []);

  useHotkeys([["n", onNewChat]]);

  return (
    <HeaderContainer className="shaded">
      <Burger />
      <h2>MemoryAgent</h2>
      <div className="spacer" />
    </HeaderContainer>
  );
}

export { Header };
