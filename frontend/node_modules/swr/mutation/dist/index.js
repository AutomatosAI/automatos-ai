Object.defineProperty(exports, '__esModule', { value: true });

var React = require('react');
var useSWR = require('swr');
var _internal = require('swr/_internal');

function _interopDefault (e) { return e && e.__esModule ? e : { default: e }; }

var React__default = /*#__PURE__*/_interopDefault(React);
var useSWR__default = /*#__PURE__*/_interopDefault(useSWR);

const startTransition = _internal.IS_REACT_LEGACY ? (cb)=>{
    cb();
} : React__default.default.startTransition;
/**
 * An implementation of state with dependency-tracking.
 */ const useStateWithDeps = (state)=>{
    const [, rerender] = React.useState({});
    const unmountedRef = React.useRef(false);
    const stateRef = React.useRef(state);
    // If a state property (data, error, or isValidating) is accessed by the render
    // function, we mark the property as a dependency so if it is updated again
    // in the future, we trigger a rerender.
    // This is also known as dependency-tracking.
    const stateDependenciesRef = React.useRef({
        data: false,
        error: false,
        isValidating: false
    });
    /**
   * @param payload To change stateRef, pass the values explicitly to setState:
   * @example
   * ```js
   * setState({
   *   isValidating: false
   *   data: newData // set data to newData
   *   error: undefined // set error to undefined
   * })
   *
   * setState({
   *   isValidating: false
   *   data: undefined // set data to undefined
   *   error: err // set error to err
   * })
   * ```
   */ const setState = React.useCallback((payload)=>{
        let shouldRerender = false;
        const currentState = stateRef.current;
        for(const _ in payload){
            const k = _;
            // If the property has changed, update the state and mark rerender as
            // needed.
            if (currentState[k] !== payload[k]) {
                currentState[k] = payload[k];
                // If the property is accessed by the component, a rerender should be
                // triggered.
                if (stateDependenciesRef.current[k]) {
                    shouldRerender = true;
                }
            }
        }
        if (shouldRerender && !unmountedRef.current) {
            rerender({});
        }
    }, []);
    _internal.useIsomorphicLayoutEffect(()=>{
        unmountedRef.current = false;
        return ()=>{
            unmountedRef.current = true;
        };
    });
    return [
        stateRef,
        stateDependenciesRef.current,
        setState
    ];
};

const mutation = ()=>(key, fetcher, config = {})=>{
        const { mutate } = useSWR.useSWRConfig();
        const keyRef = React.useRef(key);
        const fetcherRef = React.useRef(fetcher);
        const configRef = React.useRef(config);
        // Ditch all mutation results that happened earlier than this timestamp.
        const ditchMutationsUntilRef = React.useRef(0);
        const [stateRef, stateDependencies, setState] = useStateWithDeps({
            data: _internal.UNDEFINED,
            error: _internal.UNDEFINED,
            isMutating: false
        });
        const currentState = stateRef.current;
        const trigger = React.useCallback(async (arg, opts)=>{
            const [serializedKey, resolvedKey] = _internal.serialize(keyRef.current);
            if (!fetcherRef.current) {
                throw new Error('Can’t trigger the mutation: missing fetcher.');
            }
            if (!serializedKey) {
                throw new Error('Can’t trigger the mutation: missing key.');
            }
            // Disable cache population by default.
            const options = _internal.mergeObjects(_internal.mergeObjects({
                populateCache: false,
                throwOnError: true
            }, configRef.current), opts);
            // Trigger a mutation, and also track the timestamp. Any mutation that happened
            // earlier this timestamp should be ignored.
            const mutationStartedAt = _internal.getTimestamp();
            ditchMutationsUntilRef.current = mutationStartedAt;
            setState({
                isMutating: true
            });
            try {
                const data = await mutate(serializedKey, fetcherRef.current(resolvedKey, {
                    arg
                }), // We must throw the error here so we can catch and update the states.
                _internal.mergeObjects(options, {
                    throwOnError: true
                }));
                // If it's reset after the mutation, we don't broadcast any state change.
                if (ditchMutationsUntilRef.current <= mutationStartedAt) {
                    var _options_onSuccess, _options;
                    startTransition(()=>setState({
                            data,
                            isMutating: false,
                            error: undefined
                        }));
                    (_options_onSuccess = (_options = options).onSuccess) == null ? void 0 : _options_onSuccess.call(_options, data, serializedKey, options);
                }
                return data;
            } catch (error) {
                // If it's reset after the mutation, we don't broadcast any state change
                // or throw because it's discarded.
                if (ditchMutationsUntilRef.current <= mutationStartedAt) {
                    var _options_onError, _options1;
                    startTransition(()=>setState({
                            error: error,
                            isMutating: false
                        }));
                    (_options_onError = (_options1 = options).onError) == null ? void 0 : _options_onError.call(_options1, error, serializedKey, options);
                    if (options.throwOnError) {
                        throw error;
                    }
                }
            }
        }, // eslint-disable-next-line react-hooks/exhaustive-deps
        []);
        const reset = React.useCallback(()=>{
            ditchMutationsUntilRef.current = _internal.getTimestamp();
            setState({
                data: _internal.UNDEFINED,
                error: _internal.UNDEFINED,
                isMutating: false
            });
        // eslint-disable-next-line react-hooks/exhaustive-deps
        }, []);
        _internal.useIsomorphicLayoutEffect(()=>{
            keyRef.current = key;
            fetcherRef.current = fetcher;
            configRef.current = config;
        });
        // We don't return `mutate` here as it can be pretty confusing (e.g. people
        // calling `mutate` but they actually mean `trigger`).
        // And also, `mutate` relies on the useSWR hook to exist too.
        return {
            trigger,
            reset,
            get data () {
                stateDependencies.data = true;
                return currentState.data;
            },
            get error () {
                stateDependencies.error = true;
                return currentState.error;
            },
            get isMutating () {
                stateDependencies.isMutating = true;
                return currentState.isMutating;
            }
        };
    };
/**
 * A hook to define and manually trigger remote mutations like POST, PUT, DELETE and PATCH use cases.
 *
 * @link https://swr.vercel.app/docs/mutation
 * @example
 * ```jsx
 * import useSWRMutation from 'swr/mutation'
 *
 * const {
 *   data,
 *   error,
 *   trigger,
 *   reset,
 *   isMutating
 * } = useSWRMutation(key, fetcher, options?)
 * ```
 */ const useSWRMutation = _internal.withMiddleware(useSWR__default.default, mutation);

exports.default = useSWRMutation;
