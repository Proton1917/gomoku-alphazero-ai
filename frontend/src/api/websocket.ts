import { API_BASE_URL } from './client';

export interface SocketHandle {
  close: () => void;
}

interface SocketOptions<T> {
  onMessage: (data: T) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (event: Event) => void;
}

function toWebSocketUrl(path: string): string {
  const url = new URL(path, API_BASE_URL);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

export function openJsonSocket<T>(path: string, options: SocketOptions<T>): SocketHandle {
  const socket = new WebSocket(toWebSocketUrl(path));
  socket.onopen = () => options.onOpen?.();
  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data) as T;
    options.onMessage(payload);
  };
  socket.onerror = (event) => options.onError?.(event);
  socket.onclose = () => options.onClose?.();

  return {
    close: () => {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    },
  };
}
