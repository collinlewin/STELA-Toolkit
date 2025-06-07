import warnings

class _ClearWarnings:
    """
    Runtime warning handler for wrapped numerical operations.
    Provides user-facing context for common numerical warnings.
    """
    
    @staticmethod
    def run(code_block, explanation):
        """
        Provides additional information to users when runtime warnings occur.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            # always show the warning, regardless of previous runs
            warnings.simplefilter("always")
            try:
                result = code_block()
            except Exception as e:
                print(f"Exception occurred: {e}")
                return None

            # checking for only runtime-warnings
            for w in caught_warnings:
                if issubclass(w.category, RuntimeWarning):
                    print("RuntimeWarning caught:")
                    print(f"  > {w.message}")
                    print(f"{explanation}")

            return result