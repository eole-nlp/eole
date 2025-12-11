#!/usr/bin/env python
"""
Load testing script for Eole inference server with dynamic batching.
Tests both single and multiple concurrent requests with randomized input counts.
"""

import asyncio
import aiohttp
import time
import random
import statistics
from typing import List
from dataclasses import dataclass
from eole.bin import BaseBin, register_bin


@dataclass
class RequestResult:
    """Store results from a single request"""

    request_id: int
    num_inputs: int
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: str = None


class LoadTester:
    """Load testing class for the inference server"""

    def __init__(self, base_url: str, model_id: str):
        self.base_url = base_url
        self.model_id = model_id
        self.results: List[RequestResult] = []

    def generate_random_inputs(self, min_inputs: int = 1, max_inputs: int = 5) -> List[str]:
        """Generate random number of test inputs"""
        num_inputs = random.randint(min_inputs, max_inputs)
        prompts = [
            "Tell me a joke about ",
            "Write a haiku about ",
            "Explain quantum physics in terms of ",
            "What would happen if ",
            "Describe the color blue to someone who ",
            "If you could travel back in time to ",
            "The meaning of life is related to ",
            "In a world where cats ruled, ",
        ]

        subjects = [
            "programming",
            "coffee",
            "mountains",
            "the ocean",
            "space",
            "robots",
            "dragons",
            "pizza",
            "music",
            "books",
            "trees",
        ]

        inputs = []
        for _ in range(num_inputs):
            prompt = random.choice(prompts)
            subject = random.choice(subjects)
            inputs.append(f"{prompt}{subject}")

        return inputs

    async def send_text_request(
        self, session: aiohttp.ClientSession, request_id: int, min_inputs: int = 1, max_inputs: int = 5
    ) -> RequestResult:
        """Send a single text request with random number of inputs"""
        inputs = self.generate_random_inputs(min_inputs, max_inputs)
        num_inputs = len(inputs)

        payload = {
            "model": self.model_id,
            "inputs": inputs if num_inputs > 1 else inputs[0],  # Single string if only 1 input
            "max_length": 50,
            "temperature": 0.7,
        }

        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/infer", json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                await response.json()
                end_time = time.time()

                return RequestResult(
                    request_id=request_id,
                    num_inputs=num_inputs,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=response.status == 200,
                )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                num_inputs=num_inputs,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=False,
                error=str(e),
            )

    async def send_chat_request(self, session: aiohttp.ClientSession, request_id: int) -> RequestResult:
        """Send a single chat request"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Tell me something interesting about {random.choice(['science', 'history', 'technology', 'nature'])}",  # noqa: E501
            },
        ]

        payload = {"model": self.model_id, "messages": messages, "max_length": 50, "temperature": 0.7}

        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/infer", json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                await response.json()
                end_time = time.time()

                return RequestResult(
                    request_id=request_id,
                    num_inputs=1,  # Chat is always 1 input
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=response.status == 200,
                )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                num_inputs=1,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=False,
                error=str(e),
            )

    async def send_openai_request(self, session: aiohttp.ClientSession, request_id: int) -> RequestResult:
        """Send an OpenAI-compatible chat request"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"What is {random.choice(['AI', 'machine learning', 'deep learning', 'neural networks'])}?",
            },
        ]

        payload = {"model": self.model_id, "messages": messages, "max_tokens": 50, "temperature": 0.7}

        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                await response.json()
                end_time = time.time()

                return RequestResult(
                    request_id=request_id,
                    num_inputs=1,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=response.status == 200,
                )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                num_inputs=1,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=False,
                error=str(e),
            )

    async def run_concurrent_batch(
        self,
        num_requests: int,
        request_type: str = "text",
        min_inputs: int = 1,
        max_inputs: int = 5,
        stagger_ms: int = 0,
    ) -> List[RequestResult]:
        """
        Send multiple concurrent requests.

        Args:
            num_requests: Number of concurrent requests to send
            request_type: "text", "chat", or "openai"
            min_inputs: Minimum number of inputs per request (text only)
            max_inputs: Maximum number of inputs per request (text only)
            stagger_ms: Milliseconds to stagger request starts (0 = all at once)
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                if stagger_ms > 0:
                    await asyncio.sleep(stagger_ms / 1000.0)

                if request_type == "text":
                    task = self.send_text_request(session, i, min_inputs, max_inputs)
                elif request_type == "chat":
                    task = self.send_chat_request(session, i)
                elif request_type == "openai":
                    task = self.send_openai_request(session, i)
                else:
                    raise ValueError(f"Unknown request type: {request_type}")

                tasks.append(task)

            results = await asyncio.gather(*tasks)
            self.results.extend(results)
            return results

    def print_statistics(self, results: List[RequestResult] = None):
        """Print statistics from test results"""
        if results is None:
            results = self.results

        if not results:
            print("No results to analyze")
            return

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

        if successful:
            durations = [r.duration for r in successful]
            input_counts = [r.num_inputs for r in successful]

            print("\nDuration Statistics:")
            print(f"  Mean: {statistics.mean(durations):.3f}s")
            print(f"  Median: {statistics.median(durations):.3f}s")
            print(f"  Min: {min(durations):.3f}s")
            print(f"  Max: {max(durations):.3f}s")
            if len(durations) > 1:
                print(f"  StdDev: {statistics.stdev(durations):.3f}s")

            print("\nInput Count Statistics:")
            print(f"  Total inputs processed: {sum(input_counts)}")
            print(f"  Mean inputs per request: {statistics.mean(input_counts):.1f}")
            print(f"  Min inputs: {min(input_counts)}")
            print(f"  Max inputs: {max(input_counts)}")

            # Calculate throughput
            if results:
                earliest_start = min(r.start_time for r in results)
                latest_end = max(r.end_time for r in results)
                total_time = latest_end - earliest_start
                throughput = len(successful) / total_time if total_time > 0 else 0
                input_throughput = sum(input_counts) / total_time if total_time > 0 else 0

                print("\nThroughput:")
                print(f"  Requests/second: {throughput:.2f}")
                print(f"  Inputs/second: {input_throughput:.2f}")
                print(f"  Total wall time: {total_time:.3f}s")

        if failed:
            print("\nFailed requests:")
            for r in failed[:5]:  # Show first 5 failures
                print(f"  Request {r.request_id}: {r.error}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        print("=" * 60 + "\n")


@register_bin(name="load_test")
class LoadTest(BaseBin):
    """Load testing tool for Eole inference server."""

    @classmethod
    def add_args(cls, parser):
        """Add command-line arguments."""
        parser.add_argument(
            "--url", default="http://localhost:5000", help="Server URL (default: http://localhost:5000)"
        )
        parser.add_argument(
            "--model", default="Hunyuan-MT-7B-eole", help="Model ID to test (default: Hunyuan-MT-7B-eole)"
        )
        parser.add_argument(
            "--type",
            choices=["text", "chat", "openai", "mixed"],
            default="text",
            help="Request type to test (default: text)",
        )
        parser.add_argument(
            "--concurrent", type=int, default=5, help="Number of concurrent requests per batch (default: 5)"
        )
        parser.add_argument("--batches", type=int, default=3, help="Number of batches to run (default: 3)")
        parser.add_argument(
            "--min-inputs", type=int, default=1, help="Minimum inputs per request for text mode (default: 1)"
        )
        parser.add_argument(
            "--max-inputs", type=int, default=5, help="Maximum inputs per request for text mode (default: 5)"
        )
        parser.add_argument(
            "--stagger", type=int, default=0, help="Milliseconds to stagger request starts, 0=all at once (default: 0)"
        )
        parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between batches (default: 1.0)")

    @classmethod
    def run(cls, args):
        """Run the load test."""
        asyncio.run(cls._async_run(args))

    @classmethod
    async def _async_run(cls, args):
        """Async implementation of the load test."""
        tester = LoadTester(args.url, args.model)

        print(f"Starting load test against {args.url}")
        print(f"Model: {args.model}")
        print(f"Request type: {args.type}")
        print(f"Concurrent requests per batch: {args.concurrent}")
        print(f"Number of batches: {args.batches}")
        if args.type == "text":
            print(f"Inputs per request: {args.min_inputs}-{args.max_inputs} (randomized)")
        print(f"Stagger: {args.stagger}ms\n")

        for batch_num in range(args.batches):
            print(f"Running batch {batch_num + 1}/{args.batches}...")

            if args.type == "mixed":
                request_types = ["text", "chat", "openai"]
                request_type = random.choice(request_types)
            else:
                request_type = args.type

            results = await tester.run_concurrent_batch(
                num_requests=args.concurrent,
                request_type=request_type,
                min_inputs=args.min_inputs,
                max_inputs=args.max_inputs,
                stagger_ms=args.stagger,
            )

            print(f"  Completed {len([r for r in results if r.success])}/{len(results)} successfully")

            if batch_num < args.batches - 1:
                await asyncio.sleep(args.delay)

        # Print final statistics
        tester.print_statistics()
