import os


def print_section_separator(title: str, spacing: int = 2) -> None:
    """Print a section separator.

    :param title: title of the section
    :param spacing: spacing between the sections
    """
    separator_length = os.get_terminal_size().columns
    separator_char = '='
    title_char = ' '
    separator = separator_char * separator_length
    title_padding = (separator_length - len(title)) // 2
    centered_title = f"{title_char * title_padding}{title}{title_char * title_padding}" if len(
        title) % 2 == 0 else f"{title_char * title_padding}{title}{title_char * (title_padding + 1)}"
    print("\n" * spacing)
    print(f"{separator}\n{centered_title}\n{separator}")
    print("\n" * spacing)
