import time


def create_elapsed_timer(time_unit):
    tic = time.perf_counter()

    def time_units():
        return {
            'sec': lambda elapsed_frac_sec: round(elapsed_frac_sec, 3)
        }

    def elapsed():
        return '%f %s' % (time_units().get(time_unit)(time.perf_counter() - tic), time_unit)

    return elapsed
