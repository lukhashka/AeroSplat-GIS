import type { SVGProps } from "react";

const PATHS = {
  map: "M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2Zm0 0v14m6-12v14",
  jobs: "M4 6h16M4 12h10M4 18h7m9-6-3 3 3 3",
  cube: "m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 0v18m8-13.5-8 4.5-8-4.5",
  upload: "M12 16V4m0 0-4 4m4-4 4 4M5 15v4h14v-4",
  play: "M7 5v14l11-7L7 5Z",
  stop: "M7 7h10v10H7z",
  plus: "M12 5v14M5 12h14",
  external: "M14 4h6v6m0-6-9 9M18 14v5H5V6h5",
  chip: "M8 8h8v8H8zM4 10h2m-2 4h2m12-4h2m-2 4h2M10 4v2m4-2v2m-4 12v2m4-2v2",
  pin: "M12 21s7-6.2 7-11.5A7 7 0 0 0 5 9.5C5 14.8 12 21 12 21Zm0-9a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z",
  camera: "M4 8h3l2-3h6l2 3h3v11H4V8Zm8 9a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z",
  close: "M6 6l12 12M18 6 6 18",
  refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
  pulse: "M3 12h4l3-7 4 14 3-7h4",
  report: "M7 3h7l5 5v13H7V3Zm7 0v5h5M10 13h6m-6 4h6",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
