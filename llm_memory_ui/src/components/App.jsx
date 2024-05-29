import React from "react";
import { MantineProvider } from "@mantine/core";
import { ModalsProvider } from "@mantine/modals";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { LandingPage } from "./pages/LandingPage";
import { ChatPage } from "./pages/ChatPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <ChatPage />,
  },
]);

function App() {
  return (
    <MantineProvider
      theme={{ colorScheme: "dark" }}
      withGlobalStyles
      withNormalizeCSS
    >
      <ModalsProvider>
        <RouterProvider router={router} />
      </ModalsProvider>
    </MantineProvider>
  );
}

export { App };
