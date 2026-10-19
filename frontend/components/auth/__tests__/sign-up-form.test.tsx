/**
 * Clerk production readiness ("Automatos billing options", 19 Oct 2026, section 4, bug 3): in
 * waitlist mode Clerk refuses the sign-up with `sign_up_restricted_waitlist`, and the app's own
 * sign-up form showed "You're on the list" without ever joining the waitlist, so the email was
 * lost. The form now joins Clerk's waitlist with the email it was given (clerk.joinWaitlist, the
 * call Studio's waitlist form makes) and only says "on the list" once Clerk has accepted it.
 *
 * Clerk, Next's router, image and link, and framer-motion are stubbed; the form and its parts
 * are real.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'

const { createMock, prepareMock, joinWaitlistMock } = vi.hoisted(() => ({
  createMock: vi.fn(),
  prepareMock: vi.fn(),
  joinWaitlistMock: vi.fn(),
}))

vi.mock('@clerk/nextjs', () => ({
  useSignUp: () => ({
    isLoaded: true,
    signUp: {
      create: createMock,
      prepareEmailAddressVerification: prepareMock,
      attemptEmailAddressVerification: vi.fn(),
      authenticateWithRedirect: vi.fn(),
    },
    setActive: vi.fn(),
  }),
  useClerk: () => ({ joinWaitlist: joinWaitlistMock }),
}))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('next/image', () => ({
  default: (props: { alt?: string; src?: unknown }) =>
    React.createElement('img', { alt: props.alt ?? '', src: typeof props.src === 'string' ? props.src : '' }),
}))
vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) => <a href={href}>{children}</a>,
}))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
  AnimatePresence: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}))

import { SignUpForm } from '../sign-up-form'

const EMAIL = 'person@example.test'
const ON_THE_LIST = "You're on the list"
const WAITLIST_MODE = { errors: [{ code: 'sign_up_restricted_waitlist', message: 'Sign-ups are currently unavailable.' }] }

function submitSignUp(email = EMAIL) {
  fireEvent.change(screen.getByLabelText('Email'), { target: { value: email } })
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'correct-horse-battery' } })
  const form = screen.getByRole('button', { name: /join waitlist/i }).closest('form')
  if (!form) throw new Error('the sign-up form is not rendered')
  fireEvent.submit(form)
}

beforeEach(() => {
  createMock.mockReset()
  prepareMock.mockReset()
  joinWaitlistMock.mockReset()
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('SignUpForm in waitlist mode', () => {
  it("joins Clerk's waitlist with the email it was given, then says so", async () => {
    createMock.mockRejectedValue(WAITLIST_MODE)
    joinWaitlistMock.mockResolvedValue({ id: 'wle_test' })
    render(<SignUpForm />)

    submitSignUp()

    expect(await screen.findByRole('heading', { name: ON_THE_LIST })).toBeInTheDocument()
    expect(joinWaitlistMock).toHaveBeenCalledTimes(1)
    expect(joinWaitlistMock).toHaveBeenCalledWith({ emailAddress: EMAIL })
    expect(screen.getByText(`We'll notify you at ${EMAIL} when your spot is ready`)).toBeInTheDocument()
    expect(prepareMock).not.toHaveBeenCalled()
  })

  it("shows Clerk's message, and never says 'on the list', when the join fails", async () => {
    createMock.mockRejectedValue(WAITLIST_MODE)
    joinWaitlistMock.mockRejectedValue({ errors: [{ longMessage: 'Too many requests. Try again in a minute.' }] })
    render(<SignUpForm />)

    submitSignUp()

    expect(await screen.findByText('Too many requests. Try again in a minute.')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: ON_THE_LIST })).toBeNull()
  })

  it('falls back to a plain message when the join fails without one', async () => {
    createMock.mockRejectedValue(WAITLIST_MODE)
    joinWaitlistMock.mockRejectedValue(new Error('network down'))
    render(<SignUpForm />)

    submitSignUp()

    expect(await screen.findByText('Could not add you to the waitlist. Please try again.')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: ON_THE_LIST })).toBeNull()
  })
})

describe('SignUpForm outside waitlist mode', () => {
  it('goes on to email verification and never touches the waitlist', async () => {
    createMock.mockResolvedValue({})
    prepareMock.mockResolvedValue({})
    render(<SignUpForm />)

    submitSignUp()

    expect(await screen.findByRole('heading', { name: 'Verify Email' })).toBeInTheDocument()
    expect(createMock).toHaveBeenCalledWith({ emailAddress: EMAIL, password: 'correct-horse-battery' })
    expect(joinWaitlistMock).not.toHaveBeenCalled()
  })

  it('shows any other sign-up error without joining the waitlist', async () => {
    createMock.mockRejectedValue({ errors: [{ code: 'form_identifier_exists', longMessage: 'That email address is taken.' }] })
    render(<SignUpForm />)

    submitSignUp()

    expect(await screen.findByText('That email address is taken.')).toBeInTheDocument()
    expect(joinWaitlistMock).not.toHaveBeenCalled()
    expect(screen.queryByRole('heading', { name: ON_THE_LIST })).toBeNull()
  })
})
