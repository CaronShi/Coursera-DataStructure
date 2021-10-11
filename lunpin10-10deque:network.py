import sys
from collections import deque

class Packet (object): # packet_t: Packet Type
    def __init__ (self, arrival_time, processing_time):
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.finish_time = arrival_time + processing_time
        

def main (buffer_size, packets):
    myqueue = deque () # double ended queue
    for packet in packets:
        while len (myqueue) > 0 and myqueue [0].finish_time <= packet.arrival_time:
            p = myqueue.popleft () # dequeue: pop from queue
        # test whether the queue has space
        if len (myqueue) >= buffer_size:
            print (-1)
        else: 
            myqueue.append (packet) # enqueue: push to queue
            print (packet.arrival_time)
            



if __name__ == '__main__':

    argv = sys.stdin.read ().split () 
    buffer_size = int (argv [0])
    packet_count = int (argv [1])
    packets = [] 
    index = 2
    while index < len (argv):
        packets.append (
                Packet (
                    arrival_time = int (argv [index]), 
                    processing_time = int (argv [index+1])
                    )
                )
        index += 2

    main (buffer_size, packets)


