def print_section_separator(title, spacing=2, separator_length=250):
    separator_char = '='
    title_char = ' '
    separator = separator_char * separator_length
    title_padding = (separator_length - len(title)) // 2
    centered_title = f"{title_char * title_padding}{title}{title_char * title_padding}" if len(
        title) % 2 == 0 else f"{title_char * title_padding}{title}{title_char * (title_padding + 1)}"
    print("\n" * spacing)
    print(f"{separator}\n{centered_title}\n{separator}")
    print("\n" * spacing)
