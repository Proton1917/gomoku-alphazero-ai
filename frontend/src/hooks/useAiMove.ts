import { useEffect, useRef, useState } from 'react';

import { openJsonSocket, type SocketHandle } from '../api/websocket';
import type { AiMoveFrame } from '../types/game';

export function useAiMove(
  gameId: string | null,
  onFrame: (frame: AiMoveFrame) => void,
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
    socketRef.current = openJsonSocket<AiMoveFrame>(`/ws/game/${gameId}/ai-move`, {
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
      onError: () => errorRef.current('AI 搜索连接失败'),
    });
  }

  function stop() {
    socketRef.current?.close();
    socketRef.current = null;
    setActive(false);
  }

  return { active, start, stop };
}
