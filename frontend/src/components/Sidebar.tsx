import type { RefObject, MutableRefObject } from 'react';
import type { SessionListItem } from '../types';
import type { AuthUser } from '../auth';
import { getInitials } from '../auth';
import { LogoIcon, PlusIcon, SearchIcon } from './icons';

interface Props {
  sessions: SessionListItem[];
  activeId: string | null;
  authUser: AuthUser | null;
  userMenuOpen: boolean;
  userMenuRef: RefObject<HTMLDivElement> | MutableRefObject<HTMLDivElement | null>;
  onNewHypothesis: () => void;
  onSessionClick: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onOpenAuthModal: () => void;
  onToggleUserMenu: () => void;
  onLogout: () => void;
}

export default function Sidebar({
  sessions, activeId, authUser, userMenuOpen, userMenuRef,
  onNewHypothesis, onSessionClick, onDeleteSession, onOpenAuthModal, onToggleUserMenu, onLogout,
}: Props) {
  const groups = groupByBucket(sessions);

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <LogoIcon />
        <span>Hypothesis Forge</span>
      </div>

      <div className="sidebar-actions">
        <button type="button" className="primary-btn" onClick={onNewHypothesis}>
          <PlusIcon />
          <span>New hypothesis</span>
        </button>
        <div className="search-wrap">
          <SearchIcon />
          <input type="text" placeholder="Search sessions" />
        </div>
      </div>

      <nav className="sidebar-nav">
        {groups.map(([label, items]) => (
          <div key={label} className="sidebar-group">
            <div className="sidebar-group-label">{label}</div>
            {items.map((s) => {
              const isActive = s.id === activeId;
              return (
                <div key={s.id} className={`sidebar-item-wrap${isActive ? ' active' : ''}`}>
                  <button
                    type="button"
                    className={`sidebar-item${isActive ? ' active' : ''}`}
                    onClick={() => onSessionClick(s.id)}
                  >
                    {isActive && <span className="active-dot" />}
                    <span className="sidebar-item-title">{s.title}</span>
                    {s.status === 'developed' && (
                      <span className="sidebar-status-dot" title="Developed" />
                    )}
                  </button>
                  <button
                    type="button"
                    className="sidebar-delete-btn"
                    title="Delete session"
                    onClick={(e) => { e.stopPropagation(); onDeleteSession(s.id); }}
                  >
                    ✕
                  </button>
                </div>
              );
            })}
          </div>
        ))}
        {groups.length === 0 && (
          <div className="sidebar-empty">No sessions yet</div>
        )}
      </nav>

      {/* User area — bottom of sidebar */}
      <div className="sidebar-user-area" ref={userMenuRef}>
        {authUser ? (
          <>
            <button
              type="button"
              className="sidebar-user"
              onClick={onToggleUserMenu}
              title="Account options"
            >
              <div className="avatar">{getInitials(authUser)}</div>
              <div className="sidebar-user-meta">
                <div className="sidebar-user-name">{authUser.username}</div>
                <div className="sidebar-user-affil">{authUser.email}</div>
              </div>
              <span className="sidebar-user-chevron">{userMenuOpen ? '▲' : '▼'}</span>
            </button>
            {userMenuOpen && (
              <div className="user-menu">
                <button
                  type="button"
                  className="user-menu-item"
                  onClick={() => { onLogout(); }}
                >
                  Sign out
                </button>
                <button
                  type="button"
                  className="user-menu-item"
                  onClick={() => { onLogout(); onOpenAuthModal(); }}
                >
                  Switch account
                </button>
              </div>
            )}
          </>
        ) : (
          <button
            type="button"
            className="sidebar-signin-btn"
            onClick={onOpenAuthModal}
          >
            <div className="avatar avatar-anon">?</div>
            <div className="sidebar-user-meta">
              <div className="sidebar-user-name">Sign in</div>
              <div className="sidebar-user-affil">Sessions saved across devices</div>
            </div>
          </button>
        )}
      </div>
    </aside>
  );
}

function groupByBucket(sessions: SessionListItem[]): Array<[string, SessionListItem[]]> {
  const today = startOfDay(new Date());
  const yesterday = new Date(today.getTime() - 86_400_000);
  const weekAgo = new Date(today.getTime() - 7 * 86_400_000);

  const buckets: Record<string, SessionListItem[]> = {
    Today: [],
    Yesterday: [],
    'This week': [],
    Earlier: [],
  };

  for (const s of sessions) {
    const d = new Date(s.createdAt);
    if (d >= today) buckets.Today.push(s);
    else if (d >= yesterday) buckets.Yesterday.push(s);
    else if (d >= weekAgo) buckets['This week'].push(s);
    else buckets.Earlier.push(s);
  }

  return Object.entries(buckets).filter(([, items]) => items.length > 0);
}

function startOfDay(d: Date) {
  const c = new Date(d);
  c.setHours(0, 0, 0, 0);
  return c;
}
