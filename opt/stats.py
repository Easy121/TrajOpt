import pstats


def f8(x):
        return "%8.6f" % x
    
    
def func_std_string(func_name):  # match what old profile produced
    if func_name[:2] == ('~', 0):
        # special case for built-in functions
        name = func_name[2]
        if name.startswith('<') and name.endswith('>'):
            return '{%s}' % name[1:-1]
        else:
            return name
    else:
        return "%s:%d(%s)" % func_name


class ProfPrinter():
    def __init__(self) -> None:
        self.count = 0
    
    
    def print(self, prof, sort='cumtime', content=False, num=8):
        """Print the information in the cProfile class

        Args:
            prof (_type_): cProfile class
            sort (str, optional): sort the result by what, possible: cumtime, ncalls, tottime, stdname. Defaults to 'cumtime'.
            config (str, optional): preset configurations, possible: top8 or easy. Defaults to 'top8'.
        """
        self.count += 1
        stats = pstats.Stats(prof).strip_dirs().sort_stats(sort)
        
        if self.count == 1:
            print('')
        print(self.count, '. ', sep='', end=' ')
        print("%.6f seconds: " % stats.total_tt, end=' ')
        print(stats.total_calls, " function calls", sep='', end=' ')
        if stats.total_calls != stats.prim_calls:
            print("(%d primitive calls)" % stats.prim_calls, end=' ')
        print('')
        
        if content is True:
            width, list = self.get_print_list(stats, [num])
            if list:
                print('   ncalls  tottime  percall  cumtime  percall', end=' ')
                print('filename:lineno(function)')
                for func in list:
                    cc, nc, tt, ct, callers = stats.stats[func]
                    c = str(nc)
                    if nc != cc:
                        c = c + '/' + str(cc)
                    print(c.rjust(9), end=' ')
                    print(f8(tt), end=' ')
                    if nc == 0:
                        print(' '*8, end=' ')
                    else:
                        print(f8(tt/nc), end=' ')
                    print(f8(ct), end=' ')
                    if cc == 0:
                        print(' '*8, end=' ')
                    else:
                        print(f8(ct/cc), end=' ')
                    print(func_std_string(func))

        print('')
        
            
    def get_print_list(self, stats, sel_list):
        width = stats.max_name_len
        if stats.fcn_list:
            stat_list = stats.fcn_list[:]
            msg = "   Ordered by: " + stats.sort_type + '\n'
        else:
            stat_list = list(stats.stats.keys())
            msg = "   Random listing order was used\n"

        for selection in sel_list:
            stat_list, msg = stats.eval_print_amount(selection, stat_list, msg)

        count = len(stat_list)

        if not stat_list:
            return 0, stat_list
        # print(msg, file=stats.stream)  # no printing here
        if count < len(stats.stats):
            width = 0
            for func in stat_list:
                if  len(func_std_string(func)) > width:
                    width = len(func_std_string(func))
        return width+2, stat_list
