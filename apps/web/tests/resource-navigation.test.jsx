import React from 'react';
import { renderHook, waitFor, act } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { useResource } from '../src/features/product/useResource.js';
import { productApi } from '../src/features/product/api.js';
import { resetApiSession, setApiWorkspace } from '../src/lib/http/client.js';

beforeEach(() => resetApiSession());
it('reuses data on navigation and deduplicates simultaneous readers', async () => {
  const get = vi.spyOn(productApi, 'get').mockResolvedValue({ total: 7 });
  const first = renderHook(() => useResource('/analytics'));
  const second = renderHook(() => useResource('/analytics'));
  await waitFor(() => expect(first.result.current.data?.total).toBe(7));
  expect(get).toHaveBeenCalledTimes(1);
  first.unmount();
  second.unmount();
  const next = renderHook(() => useResource('/analytics'));
  expect(next.result.current.data).toEqual({ total: 7 });
  expect(next.result.current.loading).toBe(false);
  expect(get).toHaveBeenCalledTimes(1);
});
it('isolates filters and workspace switches immediately', async () => {
  vi.spyOn(productApi, 'get').mockImplementation(async (path) => ({ path }));
  setApiWorkspace('one');
  const view = renderHook(({ path }) => useResource(path), {
    initialProps: { path: '/analytics?branchId=one' },
  });
  await waitFor(() =>
    expect(view.result.current.data).toEqual({ path: '/analytics?branchId=one' }),
  );
  view.rerender({ path: '/analytics?branchId=two' });
  expect(view.result.current.data).toBeNull();
  await waitFor(() => expect(view.result.current.data?.path).toContain('two'));
  act(() => setApiWorkspace('two'));
  view.rerender({ path: '/analytics?branchId=two' });
  expect(view.result.current.data).toBeNull();
});
it('keeps loaded content while explicitly refreshing', async () => {
  let finish;
  vi.spyOn(productApi, 'get')
    .mockResolvedValueOnce({ total: 1 })
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
  const view = renderHook(() => useResource('/surveys'));
  await waitFor(() => expect(view.result.current.data?.total).toBe(1));
  act(() => view.result.current.reload());
  expect(view.result.current.data).toEqual({ total: 1 });
  expect(view.result.current.loading).toBe(false);
  await waitFor(() => expect(finish).toBeTypeOf('function'));
  await act(async () => finish({ total: 2 }));
  expect(view.result.current.data).toEqual({ total: 2 });
});
it('invalidates an active pending GET after a successful write and ignores its late result', async () => {
  const completions = [];
  vi.spyOn(productApi, 'get').mockImplementation(
    (path, signal) => new Promise((resolve) => completions.push({ resolve, signal })),
  );
  vi.spyOn(globalThis, 'fetch').mockImplementation(
    async (url) =>
      new Response(
        JSON.stringify(
          String(url).endsWith('/auth/session')
            ? { csrfToken: 'test-csrf' }
            : { survey: { id: 'saved' } },
        ),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
  );
  const view = renderHook(() => useResource('/surveys'));
  await waitFor(() => expect(completions).toHaveLength(1));
  await act(async () => {
    await productApi.send('/surveys', { title: 'Saved' });
  });
  await waitFor(() => expect(completions).toHaveLength(2));
  expect(completions[0].signal.aborted).toBe(true);
  await act(async () => completions[1].resolve({ surveys: ['saved'] }));
  await act(async () => completions[0].resolve({ surveys: ['obsolete'] }));
  expect(view.result.current.data).toEqual({ surveys: ['saved'] });
});
it('discards late results after session reset or access changes', async () => {
  const completions = [];
  vi.spyOn(productApi, 'get').mockImplementation(
    (path, signal) => new Promise((resolve) => completions.push({ resolve, signal })),
  );
  setApiWorkspace('one', 'owner');
  const view = renderHook(() => useResource('/feedback'));
  await waitFor(() => expect(completions).toHaveLength(1));
  act(() => setApiWorkspace('one', 'viewer'));
  await waitFor(() => expect(completions).toHaveLength(2));
  expect(completions[0].signal.aborted).toBe(true);
  await act(async () => completions[0].resolve({ responses: ['private'] }));
  expect(view.result.current.data).toBeNull();
  await act(async () => completions[1].resolve({ responses: ['assigned'] }));
  expect(view.result.current.data).toEqual({ responses: ['assigned'] });
  act(() => resetApiSession());
  await waitFor(() => expect(completions).toHaveLength(3));
  expect(view.result.current.data).toBeNull();
  await act(async () => completions[2].resolve({ responses: [] }));
  expect(view.result.current.data).toEqual({ responses: [] });
});
