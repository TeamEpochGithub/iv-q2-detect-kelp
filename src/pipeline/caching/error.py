class CachePipelineError(Exception):
    # Define the error message
    def __init__(self, message):
        self.message = message

    # Define the string representation
    def __str__(self):
        return self.message

    # Define the representation
    def __repr__(self):
        return self.message
