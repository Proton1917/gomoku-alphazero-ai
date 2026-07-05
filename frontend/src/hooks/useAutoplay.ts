import { useEffect, useRef, useState } from 'react';

import { openJsonSocket, type SocketHandle } from '../api/websocket';
import type { AutoplayFrame } from '../types/game';

export function useAutoplay(
  gameId: string | null,
  onFrame: (frame: AutoplayFrame) => void,
  onError: (message: string) => void,
) {
  const socketRef = useRef<SocketHandle | null>(null);
  const frameRef = useRef(onFrame);
  const errorRef = useRef(onError);
  const [active, setActive] = useState(false);

  useEffect(() => {
    frameRef.current = onFrame;
    errorRef.current = onError;
  }, [onError, onFrame]);

  useEffect(() => () => socketRef.current?.close(), []);

  useEffect(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setActive(false);
  }, [gameId]);

  function start() {
    if (!gameId || socketRef.current) {
      return;
    }
    socketRef.current = openJsonSocket<AutoplayFrame>(`/ws/game/${gameId}/autoplay`, {
      onMessage: (frame) => {
        frameRef.current(frame);
        if (frame.done) {
          socketRef.current = null;
          setActive(false);
        }
      },
      onOpen: () => setActive(true),
      onClose: () => {
        socketRef.current = null;
        setActive(false);
      },
      onError: () => errorRef.current('自动对弈连接失败'),
    });
  }

  function stop() {
    socketRef.current?.close();
    socketRef.current = null;
    setActive(false);
  }

  return { active, start, stop };
}
