function SettingsOption({
  focused,
  heading,
  description,
  children,
  collapsed,
}) {
  if (!heading || collapsed) {
    return children;
  }

  return (
    <section className={focused ? "focused" : ""}>
      {heading && <h3>{heading}</h3>}
      {description && (
        <div
          style={{
            fontSize: "90%",
            opacity: 0.9,
            marginTop: "-0.5rem",
          }}
        >
          {description}
        </div>
      )}
      {children}
    </section>
  );
}

export { SettingsOption };
