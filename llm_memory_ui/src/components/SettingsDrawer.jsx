import styled from "@emotion/styled";
import { Button, Drawer, Tabs } from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";
import { useCallback, useEffect } from "react";
import { UserOptionsTab } from "./UserOptionsTab";
import { ChatOptionsTab } from "./ChatOptionsTab";
import { UIPreferencesTab } from "./UIPreferencesTab";
import { useGlobalStore } from "../store/useGlobalStore";
import { useShallow } from "zustand/react/shallow";

const Container = styled.div`
  padding: 0.4rem 1rem 1rem 1rem;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  max-width: 100vw;
  max-height: 100vh;

  @media (max-width: 40em) {
    padding: 0;
  }

  .mantine-Tabs-root {
    display: flex;
    flex-direction: column;
    height: calc(100% - 3rem);

    @media (max-width: 40em) {
      height: calc(100% - 5rem);
    }
  }

  .mantine-Tabs-tab {
    padding: 1.2rem 1.618rem 0.8rem 1.618rem;

    @media (max-width: 40em) {
      padding: 1rem;
      span {
        display: none;
      }
    }
  }

  .mantine-Tabs-panel {
    flex-grow: 1;
    overflow-y: auto;
    overflow-x: hidden;
    min-height: 0;
    margin-left: 0;
    padding: 1.2rem 0 3rem 0;

    @media (max-width: 40em) {
      padding: 1.2rem 1rem 3rem 1rem;
    }
  }

  #save {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 0 1rem 1rem 1rem;
    opacity: 1;

    .mantine-Button-root {
      height: 3rem;
    }
  }
`;

function SettingsDrawer() {
  const { closeSettingsUI, setTab, tab, option } = useGlobalStore(
    useShallow((state) => ({
      closeSettingsUI: state.settings.closeSettingsUI,
      setTab: state.settings.setTab,
      tab: state.settings.tab,
      option: state.settings.option,
    }))
  );

  const small = useMediaQuery("(max-width: 40em)");

  const close = useCallback(() => closeSettingsUI(), [closeSettingsUI]);
  const onTabChange = useCallback((tab) => setTab(tab), [setTab]);

  useEffect(() => {
    setTimeout(() => {
      document.querySelector(".focused")?.scrollIntoView();
    }, 1000);
  }, [tab, option]);

  return (
    <Drawer
      size="50rem"
      position="right"
      opened={!!tab}
      onClose={close}
      transition="slide-left"
      transitionDuration={200}
      withCloseButton={false}
    >
      <Container>
        <Tabs value={tab} onTabChange={onTabChange} style={{ margin: "0rem" }}>
          <Tabs.List grow={small}>
            <Tabs.Tab value="chat">Chat</Tabs.Tab>
            <Tabs.Tab value="ui">UI</Tabs.Tab>
            <Tabs.Tab value="user">User</Tabs.Tab>
          </Tabs.List>
          <ChatOptionsTab />
          <UIPreferencesTab />
          <UserOptionsTab />
        </Tabs>
        <div id="save">
          <Button variant="light" fullWidth size="md" onClick={close}>
            Save and Close
          </Button>
        </div>
      </Container>
    </Drawer>
  );
}

export { SettingsDrawer };
