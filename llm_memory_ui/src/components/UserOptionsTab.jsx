import { Button, FileButton } from "@mantine/core";
import { SettingsTab } from "./SettingsTab";
import { SettingsOption } from "./SettingsOption";

function UserOptionsTab() {
  return (
    <SettingsTab name="user">
      <SettingsOption heading="Import and Export">
        <div>
          <Button
            variant="light"
            style={{
              marginRight: "1rem",
            }}
          >
            Export
          </Button>
          <FileButton accept=".json">
            {(props) => (
              <Button variant="light" {...props}>
                Import
              </Button>
            )}
          </FileButton>
        </div>
      </SettingsOption>
    </SettingsTab>
  );
}

export { UserOptionsTab };
