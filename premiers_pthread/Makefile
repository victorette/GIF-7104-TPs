CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		premiers_pthread.o

LIBS =

TARGET =	premiers_pthread

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
