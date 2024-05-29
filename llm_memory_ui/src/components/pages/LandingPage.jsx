import styled from "@emotion/styled";
import { Page } from "../Page";

const Container = styled.div`
  flex-grow: 1;
  padding-bottom: 5vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  font-family: "Work Sans", sans-serif;
  line-height: 1.7;
  gap: 1rem;
`;

function LandingPage() {
  return (
    <Page id={"landing"}>
      <Container>
        <p>Hello, how can I help you today?</p>
      </Container>
    </Page>
  );
}

export { LandingPage };
