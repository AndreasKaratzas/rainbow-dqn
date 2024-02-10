

def colorstr(options, string_args, formatter=None):
    """Usage:
    
    >>> args = ['Hello', 'World']
    >>> print(
    ...    f"My name is {colorstr(options=['red', 'underline'], string_args=args, formatter=':>10')} "
    ...    f"and I like {colorstr(options=['bold', 'cyan'], string_args=['Python'], formatter=':<10')} "
    ...    f"and {colorstr(options=['cyan'], string_args=['C++'], formatter=':^10')}\n")

    Parameters
    ----------
    options : list
        List of colors to apply to the string.
    string_args : list
        List of strings to color.
    formatter : str, optional
        Spaces to add either side of the string, eg ':>n' or ':<n' where n is the number of spaces.

    Returns
    -------
    str
        Colored, formatted, and justified string.
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {'black':          '\033[30m',  # basic colors
              'red':            '\033[31m',
              'green':          '\033[32m',
              'yellow':         '\033[33m',
              'blue':           '\033[34m',
              'magenta':        '\033[35m',
              'cyan':           '\033[36m',
              'white':          '\033[37m',
              'bright_black':   '\033[90m',  # bright colors
              'bright_red':     '\033[91m',
              'bright_green':   '\033[92m',
              'bright_yellow':  '\033[93m',
              'bright_blue':    '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan':    '\033[96m',
              'bright_white':   '\033[97m',
              'end':            '\033[0m',  # miscellaneous
              'bold':           '\033[1m',
              'underline':      '\033[4m'}

    res = []
    for substr in string_args:
        # Apply formatter if provided
        if formatter:
            # Extract alignment and width from formatter
            alignment = formatter[1]
            width = int(formatter[2:])

            # Apply the necessary padding
            if alignment == '<':
                substr = substr.ljust(width)
            elif alignment == '>':
                substr = substr.rjust(width)
            elif alignment == '^':
                substr = substr.center(width)

        #res.append(''.join(colors[x] for x in options) + substr + colors['end'])
        res.append(''.join(colors[str(x)] for x in options) + str(substr) + colors['end'])

    space_char = ''.join(colors[x] for x in options) + ' ' + colors['end']
    return space_char.join(res)
