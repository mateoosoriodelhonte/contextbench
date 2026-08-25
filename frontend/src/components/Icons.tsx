import type { JSX } from "solid-js";

type IconProps = { size?: number; stroke?: number };

export function Icon(props: IconProps & { children: JSX.Element }) {
  return <svg width={props.size ?? 18} height={props.size ?? 18} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width={props.stroke ?? 1.7} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{props.children}</svg>;
}

export const ArrowUpRight = (props: IconProps) => <Icon {...props}><path d="M7 17 17 7M8 7h9v9" /></Icon>;
export const ChevronDown = (props: IconProps) => <Icon {...props}><path d="m6 9 6 6 6-6" /></Icon>;
export const ChevronRight = (props: IconProps) => <Icon {...props}><path d="m9 6 6 6-6 6" /></Icon>;
export const Database = (props: IconProps) => <Icon {...props}><ellipse cx="12" cy="5" rx="7" ry="3" /><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7" /></Icon>;
export const FileText = (props: IconProps) => <Icon {...props}><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z" /><path d="M14 3v6h6M8 13h8M8 17h6" /></Icon>;
export const Gauge = (props: IconProps) => <Icon {...props}><path d="M4 14a8 8 0 1 1 16 0" /><path d="M12 14 16 9M7 18h10" /></Icon>;
export const Layers = (props: IconProps) => <Icon {...props}><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></Icon>;
export const Menu = (props: IconProps) => <Icon {...props}><path d="M4 7h16M4 12h16M4 17h16" /></Icon>;
export const Play = (props: IconProps) => <Icon {...props}><path d="m8 5 11 7-11 7V5Z" /></Icon>;
export const Plus = (props: IconProps) => <Icon {...props}><path d="M12 5v14M5 12h14" /></Icon>;
export const Search = (props: IconProps) => <Icon {...props}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4 4" /></Icon>;
export const Settings = (props: IconProps) => <Icon {...props}><path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" /><path d="m19 13 .1-1-.1-1 2-1.5-2-3.5-2.3 1a8 8 0 0 0-1.7-1L14.7 3h-5.4l-.3 3a8 8 0 0 0-1.7 1L5 6 3 9.5 5 11a8 8 0 0 0 0 2l-2 1.5L5 18l2.3-1a8 8 0 0 0 1.7 1l.3 3h5.4l.3-3a8 8 0 0 0 1.7-1l2.3 1 2-3.5-2-1.5Z" /></Icon>;
export const Sliders = (props: IconProps) => <Icon {...props}><path d="M4 6h16M4 12h16M4 18h16" /><circle cx="9" cy="6" r="2" fill="currentColor" stroke="none" /><circle cx="15" cy="12" r="2" fill="currentColor" stroke="none" /><circle cx="11" cy="18" r="2" fill="currentColor" stroke="none" /></Icon>;
export const Spark = (props: IconProps) => <Icon {...props}><path d="m12 3 1.3 5.7L19 10l-5.7 1.3L12 17l-1.3-5.7L5 10l5.7-1.3L12 3ZM19 16l.6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z" /></Icon>;
export const Terminal = (props: IconProps) => <Icon {...props}><path d="m5 7 5 5-5 5M12 17h7" /></Icon>;
export const X = (props: IconProps) => <Icon {...props}><path d="m6 6 12 12M18 6 6 18" /></Icon>;
