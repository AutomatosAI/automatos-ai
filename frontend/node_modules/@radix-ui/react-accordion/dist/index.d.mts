import * as React from 'react';
import React__default from 'react';
import { Primitive } from '@radix-ui/react-primitive';
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible';

type Scope<C = any> = {
    [scopeName: string]: React.Context<C>[];
} | undefined;
type ScopeHook = (scope: Scope) => {
    [__scopeProp: string]: Scope;
};
interface CreateScope {
    scopeName: string;
    (): ScopeHook;
}

type Direction = 'ltr' | 'rtl';
declare const createAccordionScope: CreateScope;
interface AccordionSingleProps extends AccordionImplSingleProps {
    type: 'single';
}
interface AccordionMultipleProps extends AccordionImplMultipleProps {
    type: 'multiple';
}
declare const Accordion: React__default.ForwardRefExoticComponent<(AccordionSingleProps | AccordionMultipleProps) & React__default.RefAttributes<HTMLDivElement>>;
interface AccordionImplSingleProps extends AccordionImplProps {
    /**
     * The controlled stateful value of the accordion item whose content is expanded.
     */
    value?: string;
    /**
     * The value of the item whose content is expanded when the accordion is initially rendered. Use
     * `defaultValue` if you do not need to control the state of an accordion.
     */
    defaultValue?: string;
    /**
     * The callback that fires when the state of the accordion changes.
     */
    onValueChange?(value: string): void;
    /**
     * Whether an accordion item can be collapsed after it has been opened.
     * @default false
     */
    collapsible?: boolean;
}
interface AccordionImplMultipleProps extends AccordionImplProps {
    /**
     * The controlled stateful value of the accordion items whose contents are expanded.
     */
    value?: string[];
    /**
     * The value of the items whose contents are expanded when the accordion is initially rendered. Use
     * `defaultValue` if you do not need to control the state of an accordion.
     */
    defaultValue?: string[];
    /**
     * The callback that fires when the state of the accordion changes.
     */
    onValueChange?(value: string[]): void;
}
type PrimitiveDivProps = React__default.ComponentPropsWithoutRef<typeof Primitive.div>;
interface AccordionImplProps extends PrimitiveDivProps {
    /**
     * Whether or not an accordion is disabled from user interaction.
     *
     * @defaultValue false
     */
    disabled?: boolean;
    /**
     * The layout in which the Accordion operates.
     * @default vertical
     */
    orientation?: React__default.AriaAttributes['aria-orientation'];
    /**
     * The language read direction.
     */
    dir?: Direction;
}
type CollapsibleProps = React__default.ComponentPropsWithoutRef<typeof CollapsiblePrimitive.Root>;
interface AccordionItemProps extends Omit<CollapsibleProps, 'open' | 'defaultOpen' | 'onOpenChange'> {
    /**
     * Whether or not an accordion item is disabled from user interaction.
     *
     * @defaultValue false
     */
    disabled?: boolean;
    /**
     * A string value for the accordion item. All items within an accordion should use a unique value.
     */
    value: string;
}
/**
 * `AccordionItem` contains all of the parts of a collapsible section inside of an `Accordion`.
 */
declare const AccordionItem: React__default.ForwardRefExoticComponent<AccordionItemProps & React__default.RefAttributes<HTMLDivElement>>;
type PrimitiveHeading3Props = React__default.ComponentPropsWithoutRef<typeof Primitive.h3>;
interface AccordionHeaderProps extends PrimitiveHeading3Props {
}
/**
 * `AccordionHeader` contains the content for the parts of an `AccordionItem` that will be visible
 * whether or not its content is collapsed.
 */
declare const AccordionHeader: React__default.ForwardRefExoticComponent<AccordionHeaderProps & React__default.RefAttributes<HTMLHeadingElement>>;
type CollapsibleTriggerProps = React__default.ComponentPropsWithoutRef<typeof CollapsiblePrimitive.Trigger>;
interface AccordionTriggerProps extends CollapsibleTriggerProps {
}
/**
 * `AccordionTrigger` is the trigger that toggles the collapsed state of an `AccordionItem`. It
 * should always be nested inside of an `AccordionHeader`.
 */
declare const AccordionTrigger: React__default.ForwardRefExoticComponent<AccordionTriggerProps & React__default.RefAttributes<HTMLButtonElement>>;
type CollapsibleContentProps = React__default.ComponentPropsWithoutRef<typeof CollapsiblePrimitive.Content>;
interface AccordionContentProps extends CollapsibleContentProps {
}
/**
 * `AccordionContent` contains the collapsible content for an `AccordionItem`.
 */
declare const AccordionContent: React__default.ForwardRefExoticComponent<AccordionContentProps & React__default.RefAttributes<HTMLDivElement>>;
declare const Root: React__default.ForwardRefExoticComponent<(AccordionSingleProps | AccordionMultipleProps) & React__default.RefAttributes<HTMLDivElement>>;
declare const Item: React__default.ForwardRefExoticComponent<AccordionItemProps & React__default.RefAttributes<HTMLDivElement>>;
declare const Header: React__default.ForwardRefExoticComponent<AccordionHeaderProps & React__default.RefAttributes<HTMLHeadingElement>>;
declare const Trigger: React__default.ForwardRefExoticComponent<AccordionTriggerProps & React__default.RefAttributes<HTMLButtonElement>>;
declare const Content: React__default.ForwardRefExoticComponent<AccordionContentProps & React__default.RefAttributes<HTMLDivElement>>;

export { Accordion, AccordionContent, type AccordionContentProps, AccordionHeader, type AccordionHeaderProps, AccordionItem, type AccordionItemProps, type AccordionMultipleProps, type AccordionSingleProps, AccordionTrigger, type AccordionTriggerProps, Content, Header, Item, Root, Trigger, createAccordionScope };
