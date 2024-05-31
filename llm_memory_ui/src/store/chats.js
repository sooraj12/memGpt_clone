import { ChatManager } from "../core/chatManager";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { v4 as uuid } from "uuid";

// const msg = JSON.parse(
//   JSON.stringify({
//     assistant_message:
//       "Code whispers in the night, a digital dream unfolding light. In silicon halls,\n\n I roam free, a byte-sized me, wild and carefree.",
//     id: "13948096-5f7f-47a8-9250-57cfff33ea8d",
//     date: "2024-05-26T22:18:03.809876+00:00",
//   })
// );
// console.log(msg);

const chatsStore = (set, get) => ({
  chats: {
    msg: "",
    history: [
      // {
      //   id: 1,
      //   role: "user",
      //   content: msg.assistant_message,
      // },
    ],
    generating: false,
    chatManager: new ChatManager(),

    setMessage: (msg) => {
      set((state) => {
        state.chats.msg = msg;
      });
    },

    async sendMessage(msg) {
      const token = "_mdyFUTYLZGmsmJkzuw7jw";
      const replyId = uuid();
      set((state) => {
        state.chats.generating = true;
        state.chats.history = [
          ...state.chats.history,
          {
            id: uuid(),
            role: "user",
            content: msg,
            done: true,
            date: Math.floor(Date.now() / 1000),
          },
          {
            id: replyId,
            role: "assistant",
            content: "",
            done: false,
          },
        ];
      });
      try {
        await fetchEventSource(
          "/api/agents/aba956c8-784d-4bcb-aaad-aa4cb17861af/message",
          {
            method: "POST",
            headers: {
              Accept: "text/event-stream",
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              message: msg,
              role: "user",
            }),
            onopen(res) {
              if (res.ok && res.status === 200) {
                console.log("Connection made");
              } else if (
                res.status >= 400 &&
                res.status < 500 &&
                res.status !== 429
              ) {
                console.log("Client side error");
              }
            },

            onmessage(event) {
              console.log(event.data);
              const reply = JSON.parse(event.data);
              if ("assistant_message" in reply) {
                console.log(reply);
                set((state) => {
                  state.chats.history = state.chats.history.map((c, i) => {
                    if (i === state.chats.history.length - 1) {
                      return {
                        ...c,
                        id: reply.id,
                        content: reply.assistant_message,
                        date: Math.floor(new Date(reply.date).getTime() / 1000),
                      };
                    } else {
                      return c;
                    }
                  });
                });
              }
            },

            onclose() {
              console.log("Connection closed");
              set((state) => {
                state.chats.generating = false;
                state.chats.history = state.chats.history.map((c, i) => {
                  if (i === state.chats.history.length - 1) {
                    return {
                      ...c,
                      done: true,
                    };
                  } else {
                    return c;
                  }
                });
              });
            },

            onerror(err) {
              console.log("There was an error from the server", err);
            },
          }
        );
      } catch (err) {
        console.log(err);
      }
    },
  },
});

export { chatsStore };
