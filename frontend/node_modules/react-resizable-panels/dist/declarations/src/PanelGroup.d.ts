import { Direction } from "./types.js";
import { CSSProperties, HTMLAttributes, PropsWithChildren } from "./vendor/react.js";
export type ImperativePanelGroupHandle = {
    getId: () => string;
    getLayout: () => number[];
    setLayout: (layout: number[]) => void;
};
export type PanelGroupStorage = {
    getItem(name: string): string | null;
    setItem(name: string, value: string): void;
};
export type PanelGroupOnLayout = (layout: number[]) => void;
export type PanelGroupProps = Omit<HTMLAttributes<keyof HTMLElementTagNameMap>, "id"> & PropsWithChildren<{
    autoSaveId?: string | null;
    className?: string;
    direction: Direction;
    id?: string | null;
    keyboardResizeBy?: number | null;
    onLayout?: PanelGroupOnLayout | null;
    storage?: PanelGroupStorage;
    style?: CSSProperties;
    tagName?: keyof HTMLElementTagNameMap;
}>;
export declare const PanelGroup: import("react").ForwardRefExoticComponent<Omit<HTMLAttributes<keyof HTMLElementTagNameMap>, "id"> & {
    autoSaveId?: string | null | undefined;
    className?: string | undefined;
    direction: Direction;
    id?: string | null | undefined;
    keyboardResizeBy?: number | null | undefined;
    onLayout?: PanelGroupOnLayout | null | undefined;
    storage?: PanelGroupStorage | undefined;
    style?: CSSProperties | undefined;
    tagName?: keyof HTMLElementTagNameMap | undefined;
} & {
    children?: import("react").ReactNode;
} & import("react").RefAttributes<ImperativePanelGroupHandle>>;
