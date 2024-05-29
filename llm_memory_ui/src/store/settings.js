const settingsStore = (set, get) => ({
  settings: {
    tab: "",
    option: "",

    setTab(tab) {
      set((state) => {
        state.settings.tab = tab;
      });
    },

    setOption(option) {
      set((state) => {
        state.settings.tab = option;
      });
    },

    setTabAndOption(tab, option) {
      set((state) => {
        state.settings.tab = tab;
        state.settings.option = option;
      });
    },

    closeSettingsUI() {
      get().settings.setTabAndOption("", "");
    },
  },
});

export { settingsStore };
