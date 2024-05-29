import styled from "@emotion/styled";
import { Header } from "./Header";
import { MessageInput } from "./MessageInput";
import { SettingsDrawer } from "./SettingsDrawer";

const Container = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: row;
  overflow: hidden;

  background: #292933;
  color: white;

  .sidebar {
    width: 0%;
    height: 100%;
    background: #303038;
    flex-shrink: 0;

    @media (min-width: 40em) {
      transition: width 0.2s ease-in-out;
    }

    &.opened {
      width: 33.33%;

      @media (max-width: 40em) {
        width: 100%;
        flex-shrink: 0;
      }

      @media (min-width: 50em) {
        width: 25%;
      }

      @media (min-width: 60em) {
        width: 20%;
      }
    }
  }

  @media (max-width: 40em) {
    .sidebar.opened + div {
      display: none;
    }
  }
`;

const Main = styled.div`
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  overflow: scroll;

  @media (min-height: 30em) {
    overflow: hidden;
  }
`;

function Page({ id, children }) {
  return (
    <Container>
      <Main key={id}>
        <Header />
        {children}

        <MessageInput key={1} />
        <SettingsDrawer />
      </Main>
    </Container>
  );
}

export { Page };
