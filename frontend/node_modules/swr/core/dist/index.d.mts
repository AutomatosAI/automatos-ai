import * as swr__internal from 'swr/_internal';
import { Key, FullConfiguration, SWRHook } from 'swr/_internal';
export { Arguments, BareFetcher, Cache, Fetcher, Key, KeyLoader, KeyedMutator, Middleware, MutatorCallback, MutatorOptions, Revalidator, RevalidatorOptions, SWRConfiguration, SWRHook, SWRResponse, State, mutate, preload, useSWRConfig } from 'swr/_internal';
import ReactExports from 'react';

declare const unstable_serialize: (key: Key) => string;

declare const SWRConfig: ReactExports.FC<ReactExports.PropsWithChildren<{
    value?: (Partial<swr__internal.PublicConfiguration<any, any, swr__internal.BareFetcher<any>>> & Partial<swr__internal.ProviderConfiguration> & {
        provider?: ((cache: Readonly<swr__internal.Cache<any>>) => swr__internal.Cache<any>) | undefined;
    }) | ((parentConfig?: (Partial<swr__internal.PublicConfiguration<any, any, swr__internal.BareFetcher<any>>> & Partial<swr__internal.ProviderConfiguration> & {
        provider?: ((cache: Readonly<swr__internal.Cache<any>>) => swr__internal.Cache<any>) | undefined;
    }) | undefined) => Partial<swr__internal.PublicConfiguration<any, any, swr__internal.BareFetcher<any>>> & Partial<swr__internal.ProviderConfiguration> & {
        provider?: ((cache: Readonly<swr__internal.Cache<any>>) => swr__internal.Cache<any>) | undefined;
    }) | undefined;
}>> & {
    defaultValue: FullConfiguration;
};

/**
 * A hook to fetch data.
 *
 * @link https://swr.vercel.app
 * @example
 * ```jsx
 * import useSWR from 'swr'
 * function Profile() {
 *   const { data, error, isLoading } = useSWR('/api/user', fetcher)
 *   if (error) return <div>failed to load</div>
 *   if (isLoading) return <div>loading...</div>
 *   return <div>hello {data.name}!</div>
 * }
 * ```
 */
declare const useSWR: SWRHook;

export { SWRConfig, useSWR as default, unstable_serialize };
