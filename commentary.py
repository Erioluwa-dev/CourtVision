
class CommentaryEngine:
    """
    Generates basketball commentary
    from CourtVision analytics.
    """

    def __init__(self):

        self.events = []

    def add_event(
        self,
        text,
    ):
        """
        Store a commentary event.
        """

        self.events.append(text)

    def get_events(
        self,
    ):
        """
        Return every commentary event.
        """

        return self.events

    def latest_event(
        self,
    ):
        """
        Return the newest event.
        """

        if len(self.events) == 0:
            return None

        return self.events[-1]
