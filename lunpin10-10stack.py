import sys


class MyStack (list):
    def __init__ (self):
        super ().__init__ ()


def main ():
    argv = sys.stdin.read ().split ()
    # below is a tokenizer routine
    if len (argv) <= 1: return # no queries
    stack = MyStack ()
    # [5 push 2 push 1 max pop max]
    tokenIndex = 1
    while tokenIndex < len (argv):
        token = argv [tokenIndex]
        if token == 'max':
            print (max (stack))
        elif token == 'pop':
            stack.pop () 
        elif token == 'push':
            tokenIndex += 1
            value = int (argv [tokenIndex])
            stack.append (value)
        else: raise Exception () # command not found
        tokenIndex += 1
        
    i = 1
    match i:
        case 1: print ('hello')
        case 2: print ('hhello')
        case _: print ('_ello')


if __name__ == '__main__':
    main () 
