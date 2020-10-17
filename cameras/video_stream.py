# import the necessary packages
from threading import Thread
import cv2

class VideoStream:
    """ Tạo thread kéo stream với opencv
    Attributes:
        src: Link rtsp (default: 0-webcam)
        name: Tên thread
    """
    def __init__(self, src=0, name="VideoStream"):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        self.name = str(name)
        self.stopped = False
        self.reconnected = False
        self.updated = False


    def start(self):
        """Start luồng stream với threading
        
        Returns:
            VideoStream obj -- Đối tượng video stream
        """
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Tự động cập nhật (kéo) frame
        """
        while True:
            if self.stopped:
                return
            elif self.reconnected:
                self.stream.release()
                del self.stream
                self.stream = cv2.VideoCapture(self.src)
                self.reconnected = False

            (grabbed, frame) = self.stream.read()
            if not self.updated:
                self.updated = True
                self.grabbed, self.frame = grabbed, frame


    def read(self, override=True):
        """Trả ra frame nếu chưa đc đọc
        Keyword Arguments:
            override {bool} -- ép đọc frame (default: {False})
        Returns:
            numpy array -- frame
        """
        if self.updated:
            self.updated = False
            return self.frame

        elif override:
            return self.frame


    def stop(self):
        """Dừng luồng stream
        """
        self.stopped = True
    

    def reconnect(self, src=None):
        """Kết nối lại luồng
        
        Keyword Arguments:
            src {str} -- Rtsp link stream (default: {None})
        """
        self.reconnected = True
        if src is not None:
            self.src = src
