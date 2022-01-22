
def parse_leap_seconds_file(filename: str):
    """[summary]

    Args:
        filename (str): [description]
    """
    #TODO check filename validity
    with open(filename, mode="r") as f:
        print(f.readlines())
