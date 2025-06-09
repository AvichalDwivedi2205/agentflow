# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supabase client utilities for real-time database operations."""

import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
from supabase import create_client, Client
from realtime import RealtimeClient, RealtimeChannel
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class SupabaseManager:
    """Manages Supabase client connections and real-time subscriptions."""
    
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key
        )
        self.realtime_client: Optional[RealtimeClient] = None
        self.channels: Dict[str, RealtimeChannel] = {}
        self._connected = False
    
    async def connect_realtime(self):
        """Initialize real-time connection."""
        if not self._connected:
            self.realtime_client = RealtimeClient(
                settings.supabase_url.replace('https://', 'wss://') + '/realtime/v1',
                settings.supabase_anon_key
            )
            await self.realtime_client.connect()
            self._connected = True
            logger.info("Connected to Supabase Realtime")
    
    async def disconnect_realtime(self):
        """Disconnect from real-time."""
        if self.realtime_client and self._connected:
            await self.realtime_client.disconnect()
            self._connected = False
            logger.info("Disconnected from Supabase Realtime")
    
    def subscribe_to_table_changes(
        self,
        table: str,
        callback: Callable[[Dict[str, Any]], None],
        event_type: str = "*"
    ) -> str:
        """Subscribe to table changes via Postgres Changes."""
        if not self.realtime_client:
            raise RuntimeError("Realtime client not connected")
        
        channel_name = f"table_{table}_{event_type}"
        channel = self.realtime_client.channel(channel_name)
        
        channel.on_postgres_changes(
            event=event_type,
            schema="public",
            table=table,
            callback=callback
        )
        
        channel.subscribe()
        self.channels[channel_name] = channel
        
        logger.info(f"Subscribed to {table} table changes")
        return channel_name
    
    def subscribe_to_broadcast(
        self,
        channel_name: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """Subscribe to broadcast messages."""
        if not self.realtime_client:
            raise RuntimeError("Realtime client not connected")
        
        channel = self.realtime_client.channel(channel_name)
        channel.on_broadcast(event="message", callback=callback)
        channel.subscribe()
        
        self.channels[channel_name] = channel
        logger.info(f"Subscribed to broadcast channel: {channel_name}")
        return channel_name
    
    async def broadcast_message(
        self,
        channel_name: str,
        message: Dict[str, Any]
    ):
        """Send broadcast message to channel."""
        if channel_name in self.channels:
            await self.channels[channel_name].send_broadcast(
                event="message",
                payload=message
            )
            logger.debug(f"Broadcast message sent to {channel_name}")
    
    def unsubscribe(self, channel_name: str):
        """Unsubscribe from a channel."""
        if channel_name in self.channels:
            self.channels[channel_name].unsubscribe()
            del self.channels[channel_name]
            logger.info(f"Unsubscribed from channel: {channel_name}")
    
    async def insert_record(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Insert a record into a table."""
        try:
            result = self.client.table(table).insert(data).execute()
            logger.debug(f"Inserted record into {table}")
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error inserting into {table}: {e}")
            raise
    
    async def update_record(
        self,
        table: str,
        record_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a record in a table."""
        try:
            result = self.client.table(table).update(data).eq("id", record_id).execute()
            logger.debug(f"Updated record {record_id} in {table}")
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error updating {table}: {e}")
            raise
    
    async def get_record(
        self,
        table: str,
        record_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        try:
            result = self.client.table(table).select("*").eq("id", record_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting record from {table}: {e}")
            return None
    
    async def query_records(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query records with optional filters."""
        try:
            query = self.client.table(table).select("*")
            
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error querying {table}: {e}")
            return []


# Global Supabase manager instance
supabase_manager = SupabaseManager() 