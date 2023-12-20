class TargetPipeline():
    def __init__(self, raw_target_path: str, processed_path: str, transformation_steps: list, column_steps: list):
        self.raw_target_path = raw_target_path
        self.processed_path = processed_path
        self.transformation_steps = transformation_steps
        self.column_steps = column_steps

    def get_pipeline(self):
        # TODO implement target pipeline
        return None
