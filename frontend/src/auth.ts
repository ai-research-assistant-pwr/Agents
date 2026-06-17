export interface AuthUser {
  id: string;
  username: string;
  email: string;
}

const TOKEN_KEY = 'hf_auth_token';
const USER_KEY = 'hf_auth_user';

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function saveAuth(token: string, user: AuthUser): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function getAuthUser(): AuthUser | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw) as AuthUser; } catch { return null; }
}

export function getAuthHeader(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function getInitials(user: AuthUser): string {
  return user.username.slice(0, 2).toUpperCase();
}
