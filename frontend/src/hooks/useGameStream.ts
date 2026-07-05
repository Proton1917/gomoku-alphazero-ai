import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { openJsonSocket, type SocketHandle } from '../api/websocket';

interface StreamOptions<T> {
  onFrame: (frame: T) => void;
  onError: (message: string) => void;
  /** 连接结束时回调；receivedDone 表示是否收到过 done 帧（干净结束） */
  onEnd?: (receivedDone: boolean) => void;
}

/**
 * 通用的对局 WebSocket 流（autoplay / ai-move 共用）。
 *
 * 关键点：所有回调都校验句柄身份（socketRef.current === handle），
 * 旧连接的迟到事件不会影响新连接的状态。
 */
export function useGameStream<T extends { done: boolean }>(
  path: string | null,
  errorMessage: string,
  options: StreamOptions<T>,
) {
  const socketRef = useRef<SocketHandle | null>(null);
  const optionsRef = useRef(options);
  const [active, setActive] = useState(false);

  useEffect(() => {
    optionsRef.current = options;
  });

  const stop = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    setActive(false);
  }, []);

  // path 变化或组件卸载时关闭当前连接
  useEffect(() => stop, [path, stop]);

  const start = useCallback(() => {
    if (!path || socketRef.current) {
      return;
    }
    let receivedDone = false;
    const handle = openJsonSocket<T>(path, {
      onMessage: (frame) => {
        optionsRef.current.onFrame(frame);
        if (frame.done) {
          receivedDone = true;
          if (socketRef.current === handle) {
            socketRef.current = null;
            setActive(false);
          }
          handle.close();
          optionsRef.current.onEnd?.(true);
        }
      },
      onClose: () => {
        if (socketRef.current === handle) {
          socketRef.current = null;
          setActive(false);
          optionsRef.current.onEnd?.(receivedDone);
        }
      },
      onError: () => {
        if (socketRef.current === handle) {
          optionsRef.current.onError(errorMessage);
        }
      },
    });
    socketRef.current = handle;
    setActive(true);
  }, [path, errorMessage]);

  return useMemo(() => ({ active, start, stop }), [active, start, stop]);
}
